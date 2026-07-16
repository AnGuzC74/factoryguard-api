import os
import sys
import tomllib
from pathlib import Path
import numpy as np
import polars as pl
import streamlit as st
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))
from core.dsp import (
    ejecutar_fft,
    calcular_rms,
    calcular_punto_inflexion_log,
    calcular_rul_hibrido
)

st.set_page_config(page_title="Industrial AI - Visor de Datos", page_icon="📊", layout="wide")

# ============================================================
# FUNCIÓN DE FORMATO DE TIEMPO INTELIGENTE
# ============================================================

def formato_tiempo_legible(horas: float) -> str:
    """
    Convierte horas en una cadena legible con la unidad más adecuada.
    Ejemplos:
        - 0.0125 horas → 45 segundos
        - 0.0478 horas → 2 minutos 52 segundos
        - 4.08 horas → 4 horas 5 minutos
        - 28.5 horas → 1 día 4 horas 30 minutos
    """
    if horas is None or horas < 0:
        return "Sin proyección"
    
    # Convertir a segundos
    total_segundos = horas * 3600.0
    
    if total_segundos < 60:
        return f"{int(total_segundos)} segundos"
    elif total_segundos < 3600:
        minutos = int(total_segundos // 60)
        segundos = int(total_segundos % 60)
        return f"{minutos} minuto{'s' if minutos != 1 else ''} {segundos} segundo{'s' if segundos != 1 else ''}"
    elif total_segundos < 86400:  # < 24 horas
        horas_enteras = int(total_segundos // 3600)
        minutos = int((total_segundos % 3600) // 60)
        if minutos == 0:
            return f"{horas_enteras} hora{'s' if horas_enteras != 1 else ''}"
        return f"{horas_enteras} hora{'s' if horas_enteras != 1 else ''} {minutos} minuto{'s' if minutos != 1 else ''}"
    else:
        dias = int(total_segundos // 86400)
        resto = total_segundos % 86400
        horas_resto = int(resto // 3600)
        minutos_resto = int((resto % 3600) // 60)
        if horas_resto == 0 and minutos_resto == 0:
            return f"{dias} día{'s' if dias != 1 else ''}"
        elif minutos_resto == 0:
            return f"{dias} día{'s' if dias != 1 else ''} {horas_resto} hora{'s' if horas_resto != 1 else ''}"
        else:
            return f"{dias} día{'s' if dias != 1 else ''} {horas_resto} hora{'s' if horas_resto != 1 else ''} {minutos_resto} minuto{'s' if minutos_resto != 1 else ''}"


# ============================================================
# CLASE PRINCIPAL DEL DASHBOARD
# ============================================================

class DashboardPrognosisIndustrial:
    def __init__(self, ruta_config: Path = Path("config.toml")):
        with open(ruta_config, "rb") as f:
            self.config = tomllib.load(f)

        self.archivo_parquet = Path(self.config["infraestructura"]["archivo_parquet"])
        self.frecuencia_muestreo = self.config["fisica_rodamiento"]["frecuencia_muestreo"]
        self.puntos_fft = self.config["fisica_rodamiento"]["puntos_fft"]
        self.archivo_telemetria = Path("datos/telemetria_optimizacion.csv")

        self.UMBRAL_CRITICO = self.config["umbrales_severidad"]["critico_rms"]
        self.UMBRAL_ALERTA = self.config["umbrales_severidad"]["alerta_rms"]
        self.UMBRAL_APAGADO = self.config["umbrales_severidad"]["motor_apagado_rms"]
        self.MINUTOS_POR_CICLO = self.config["diagnostico"]["minutos_por_ciclo"]
        self.TOLERANCIA_ARMONICA = self.config["diagnostico"]["tolerancia_armonica_hz"]

        self.FREQ_BPFO = 236.4
        self.FREQ_BPFI = 296.8
        self.FREQ_BSF = 139.2
        self.FREQ_FTF = 14.8

    @st.cache_data
    def cargar_telemetria(_self, mtime: float) -> pl.DataFrame:
        if not _self.archivo_telemetria.exists():
            return pl.DataFrame()
        return pl.read_csv(_self.archivo_telemetria)

    def _es_motor_apagado(self, df_hist: pl.DataFrame) -> bool:
        ventana = self.config["diagnostico"]["ventana_persistencia_apagado"]
        if df_hist.height < ventana:
            return False
        ultimos = df_hist.tail(ventana)
        return ultimos["vibracion_rms"].mean() <= self.UMBRAL_APAGADO

    def calcular_rul_pronostico(self, df_hist: pl.DataFrame, max_rms_historico: float):
        if max_rms_historico >= self.UMBRAL_CRITICO:
            return 0.0, np.array([]), np.array([]), "Crítico (Reemplazo)"

        df_activos = df_hist.filter(pl.col("vibracion_rms") > self.UMBRAL_APAGADO)
        if df_activos.height < 15:
            return 999.0, np.array([]), np.array([]), "Datos insuficientes"

        x_ciclos = df_activos["indice_secuencial"].to_numpy()
        y_rms = df_activos["vibracion_rms"].to_numpy()

        ciclos_restantes, x_futuro, y_proyectado, modelo = calcular_rul_hibrido(
            x_ciclos, y_rms, self.UMBRAL_CRITICO, horizonte_prediccion=300
        )
        return ciclos_restantes, x_futuro, y_proyectado, modelo

    def verificar_armonicos(self, freq_dom: float, freq_base: float, max_orden: int = 5) -> bool:
        for orden in range(1, max_orden + 1):
            if abs(freq_dom - (freq_base * orden)) <= self.TOLERANCIA_ARMONICA:
                return True
        return False

    def renderizar_interfaz(self):
        # --- NAVEGACIÓN EN EL SIDEBAR ---
        st.sidebar.title("🧭 Navegación")
        pagina = st.sidebar.radio(
            "Ir a:",
            ["📊 Dashboard", "💬 Chat RAG"]
        )

        if pagina == "💬 Chat RAG":
            # Importar y ejecutar la página de chat
            try:
                from pages.chat import main as chat_main
                chat_main()
                st.stop()
            except ImportError:
                st.error("La página de chat no está disponible. Asegúrate de que existe src/app/pages/chat.py")
                st.stop()

        # Si no es Chat RAG, continuar con el Dashboard
        st.title("📊 Visor de Datos de Pronóstico Industrial")
        st.caption("Análisis objetivo de vibraciones - NASA IMS Dataset")
        st.markdown("---")

        mtime = os.path.getmtime(self.archivo_telemetria) if self.archivo_telemetria.exists() else 0.0
        df_telemetria = self.cargar_telemetria(mtime)
        if df_telemetria.is_empty():
            st.error("Ejecute primero el pipeline de ingesta y análisis.")
            return

        st.sidebar.header("🕹️ Control Temporal")
        archivos = df_telemetria["archivo_origen"].to_list()
        seleccion = st.sidebar.select_slider("Seleccione el instante:", options=archivos, value=archivos[-1])

        datos_fila = df_telemetria.filter(pl.col("archivo_origen") == seleccion).to_dicts()[0]
        idx_actual = datos_fila["indice_secuencial"]

        df_hasta_ahora = df_telemetria.filter(pl.col("indice_secuencial") <= idx_actual)
        max_rms_historico = df_hasta_ahora["vibracion_rms"].max()

        rms_actual = datos_fila["vibracion_rms"]
        freq_dom = datos_fila["frecuencia_dominante_hz"]

        motor_apagado = self._es_motor_apagado(df_hasta_ahora)

        ciclos_rul, x_fut, y_proy, modelo_sel = self.calcular_rul_pronostico(df_hasta_ahora, max_rms_historico)

        if max_rms_historico >= self.UMBRAL_CRITICO:
            horas_rul = 0.0
            ciclos_rul = 0.0
        else:
            horas_rul = (ciclos_rul * self.MINUTOS_POR_CICLO) / 60.0
            if ciclos_rul >= 999:
                horas_rul = None

        # Mostrar RUL en formato legible
        rul_legible = formato_tiempo_legible(horas_rul) if horas_rul is not None else "Sin proyección"

        zona_falla = "Ninguna"
        if not motor_apagado and max_rms_historico >= self.UMBRAL_ALERTA:
            if self.verificar_armonicos(freq_dom, self.FREQ_BPFO) or freq_dom > 3500:
                zona_falla = "Pista Externa (BPFO)"
            elif self.verificar_armonicos(freq_dom, self.FREQ_BPFI):
                zona_falla = "Pista Interna (BPFI)"
            elif self.verificar_armonicos(freq_dom, self.FREQ_BSF):
                zona_falla = "Elementos Rodantes (BSF)"
            elif self.verificar_armonicos(freq_dom, self.FREQ_FTF):
                zona_falla = "Jaula (FTF)"
            else:
                zona_falla = "Pista Externa (Modulación)"

        y_rms_array = df_hasta_ahora["vibracion_rms"].to_numpy()
        punto_micro = calcular_punto_inflexion_log(
            df_hasta_ahora["indice_secuencial"].to_numpy(),
            y_rms_array
        )

        # --- MÉTRICAS ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMS Actual", f"{rms_actual:.4f} g")
        with col2:
            st.metric("RMS Máximo Histórico", f"{max_rms_historico:.4f} g")
        with col3:
            st.metric("Frecuencia Dominante", f"{freq_dom:.1f} Hz")
        with col4:
            if motor_apagado:
                rul_text = "Motor detenido"
            elif horas_rul is None:
                rul_text = "Sin proyección"
            elif horas_rul == 0.0:
                rul_text = "0.0 horas (Reemplazo)"
            else:
                rul_text = rul_legible
            st.metric("RUL Estimado", rul_text)

        st.markdown("---")

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Modelo de regresión", modelo_sel)
        with col_info2:
            st.metric("Zona de falla detectada", zona_falla)
        with col_info3:
            if punto_micro is not None:
                st.metric("Microfisura estimada (ciclo)", str(punto_micro))
            else:
                st.metric("Microfisura estimada", "No detectada")

        st.markdown("---")

        # --- GRÁFICOS ---
        col_graf_izq, col_graf_der = st.columns(2)

        with col_graf_izq:
            st.subheader("📈 Curva de Degradación")
            fig_rul = go.Figure()
            fig_rul.add_trace(go.Scatter(
                x=df_telemetria["indice_secuencial"], y=df_telemetria["vibracion_rms"],
                mode='lines', name='Histórico completo', line=dict(color='rgba(255,255,255,0.15)', width=1)
            ))
            fig_rul.add_trace(go.Scatter(
                x=df_hasta_ahora["indice_secuencial"], y=df_hasta_ahora["vibracion_rms"],
                mode='lines', name='Histórico hasta ahora', line=dict(color='#636EFA', width=2.5)
            ))
            if not motor_apagado and len(x_fut) > 0 and ciclos_rul < 999:
                fig_rul.add_trace(go.Scatter(
                    x=x_fut, y=y_proy,
                    mode='lines', name=f'Proyección ({modelo_sel})', line=dict(color='#EF553B', dash='dash')
                ))
            fig_rul.add_hline(y=self.UMBRAL_CRITICO, line_dash="dash", line_color="orange",
                             annotation_text=f"Umbral Crítico ({self.UMBRAL_CRITICO:.2f} g)")

            fig_rul.update_layout(
                template="plotly_dark", height=450,
                xaxis_title="Ciclos", yaxis_title="RMS (g)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_rul, use_container_width=True)

        with col_graf_der:
            st.subheader("🔬 Espectro de Frecuencia")
            if self.archivo_parquet.exists():
                df_bloque = pl.read_parquet(self.archivo_parquet).filter(pl.col("archivo_origen") == seleccion)
                senal = df_bloque["rodamiento_1"].head(self.puntos_fft).to_numpy()
                freqs, amps = ejecutar_fft(senal, self.frecuencia_muestreo)

                fig_fft = go.Figure()
                fig_fft.add_trace(go.Scatter(
                    x=freqs, y=amps, mode='lines',
                    name='FFT', line=dict(color='#00CC96')
                ))
                if freq_dom > 0:
                    fig_fft.add_vline(x=freq_dom, line_dash="dash", line_color="red",
                                      annotation_text=f"{freq_dom:.1f} Hz")

                fig_fft.update_layout(
                    template="plotly_dark", height=450,
                    xaxis_title="Frecuencia (Hz)", yaxis_title="Amplitud (g)",
                    xaxis=dict(range=[0, 5000])
                )
                st.plotly_chart(fig_fft, use_container_width=True)

        st.markdown("---")

        # --- REPORTE ---
        st.subheader("📋 Reporte Técnico")

        lineas = []
        lineas.append(f"**Archivo analizado:** {seleccion}")
        lineas.append(f"**RMS actual:** {rms_actual:.4f} g")
        lineas.append(f"**RMS máximo histórico:** {max_rms_historico:.4f} g")
        lineas.append(f"**Frecuencia dominante:** {freq_dom:.1f} Hz")

        if motor_apagado:
            lineas.append("**Estado del motor:** Detenido / Apagado")
        elif horas_rul is None:
            lineas.append("**RUL estimado:** Sin proyección (tendencia estable)")
        elif horas_rul == 0.0:
            lineas.append("**RUL estimado:** 0.0 horas (Reemplazo inmediato requerido)")
        else:
            lineas.append(f"**RUL estimado:** {rul_legible}")

        lineas.append(f"**Modelo utilizado:** {modelo_sel}")
        lineas.append(f"**Zona de falla detectada:** {zona_falla}")

        if punto_micro is not None:
            lineas.append(f"**Microfisura estimada en ciclo:** {punto_micro}")

        if max_rms_historico >= self.UMBRAL_CRITICO:
            lineas.append("**Recomendación:** Reemplazo inmediato del rodamiento.")
        elif max_rms_historico >= self.UMBRAL_ALERTA:
            lineas.append("**Recomendación:** Programar mantenimiento en el corto plazo.")
        else:
            lineas.append("**Recomendación:** Operación normal, continuar monitoreo.")

        st.markdown("\n".join(lineas))


if __name__ == "__main__":
    app = DashboardPrognosisIndustrial()
    app.renderizar_interfaz()
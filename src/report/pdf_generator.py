"""
Generador de reportes PDF ejecutivos con timestamp y gráficos.
Utiliza ReportLab para el layout y Matplotlib para gráficos incrustados.
El RUL se muestra en formato de tiempo legible (segundos, minutos, horas, días).
Corregido: Ahora usa el mismo cálculo de RUL que el dashboard y el menú.
"""
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import polars as pl
import tomllib
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

sys.path.append(str(Path(__file__).parent.parent))
from core.dsp import calcular_rul_hibrido, ejecutar_fft

CONFIG_PATH = Path("config.toml")
with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

UMBRAL_CRITICO = CONFIG["umbrales_severidad"]["critico_rms"]
UMBRAL_ALERTA = CONFIG["umbrales_severidad"]["alerta_rms"]
MINUTOS_POR_CICLO = CONFIG["diagnostico"]["minutos_por_ciclo"]
FRECUENCIA_MUESTREO = CONFIG["fisica_rodamiento"]["frecuencia_muestreo"]
PUNTOS_FFT = CONFIG["fisica_rodamiento"]["puntos_fft"]

TELEMETRIA_PATH = Path("datos/telemetria_optimizacion.csv")
PARQUET_PATH = Path("datos/nasa_bearing_consolidado.parquet")

df_telemetria = pl.read_csv(TELEMETRIA_PATH)


def formato_tiempo_legible(horas: float) -> str:
    if horas is None or horas < 0:
        return "Sin proyección"
    total_segundos = horas * 3600.0
    if total_segundos < 60:
        return f"{int(total_segundos)} segundos"
    elif total_segundos < 3600:
        minutos = int(total_segundos // 60)
        segundos = int(total_segundos % 60)
        return f"{minutos} minuto{'s' if minutos != 1 else ''} {segundos} segundo{'s' if segundos != 1 else ''}"
    elif total_segundos < 86400:
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


def generar_reporte_pdf(archivo_origen: str, output_dir: Path = Path("reports")):
    """
    Genera un PDF ejecutivo con diagnóstico, gráficos y tabla de métricas.
    El estado se basa en max_rms_historico (irreversible).
    El RUL se calcula usando el histórico HASTA el archivo seleccionado.
    """
    output_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{archivo_origen}_{timestamp_str}.pdf"

    # --- DATOS DEL ARCHIVO SELECCIONADO ---
    df_fila = df_telemetria.filter(pl.col("archivo_origen") == archivo_origen)
    if df_fila.is_empty():
        raise ValueError(f"Archivo {archivo_origen} no encontrado")

    datos = df_fila.to_dicts()[0]
    idx_actual = datos["indice_secuencial"]
    rms_actual = datos["vibracion_rms"]
    freq_dom = datos["frecuencia_dominante_hz"]
    amp_max = datos["amplitud_maxima_g"]

    # --- HISTÓRICO HASTA EL ARCHIVO SELECCIONADO ---
    df_hist = df_telemetria.filter(pl.col("indice_secuencial") <= idx_actual)
    max_rms_historico = df_hist["vibracion_rms"].max()

    # --- CALCULAR RUL USANDO EL HISTÓRICO COMPLETO HASTA EL ARCHIVO ---
    x_ciclos = df_hist["indice_secuencial"].to_numpy()
    y_rms = df_hist["vibracion_rms"].to_numpy()
    
    # Usar la misma función que el dashboard y el menú
    ciclos_rul, x_fut, y_proy, modelo = calcular_rul_hibrido(
        x_ciclos, y_rms, UMBRAL_CRITICO, horizonte_prediccion=300
    )
    horas_rul = (ciclos_rul * MINUTOS_POR_CICLO) / 60.0 if ciclos_rul < 999 else None

    # --- ZONA DE FALLA ---
    def verificar_armonicos(freq, base, max_orden=5):
        for orden in range(1, max_orden + 1):
            if abs(freq - (base * orden)) <= 2.0:
                return True
        return False

    zona_falla = "No detectada"
    if max_rms_historico >= UMBRAL_ALERTA:
        if verificar_armonicos(freq_dom, 236.4) or freq_dom > 3500:
            zona_falla = "Pista Externa (BPFO)"
        elif verificar_armonicos(freq_dom, 296.8):
            zona_falla = "Pista Interna (BPFI)"
        elif verificar_armonicos(freq_dom, 139.2):
            zona_falla = "Elementos Rodantes (BSF)"
        elif verificar_armonicos(freq_dom, 14.8):
            zona_falla = "Jaula (FTF)"
        else:
            zona_falla = "Pista Externa (Modulación)"

    # --- ESTADO BASADO EN max_rms_historico (IRREVERSIBLE) ---
    if max_rms_historico >= UMBRAL_CRITICO:
        estado = "🔴 CRÍTICO - Reemplazo inmediato"
        color_estado = "red"
    elif max_rms_historico >= UMBRAL_ALERTA:
        if horas_rul is not None and horas_rul < 8:
            estado = "🟠 ALERTA AVANZADA"
            color_estado = "orange"
        elif horas_rul is not None and horas_rul < 50:
            estado = "🟡 ALERTA INCIPIENTE"
            color_estado = "gold"
        else:
            estado = "🟣 VIGILANCIA"
            color_estado = "purple"
    else:
        estado = "🟢 NORMAL - Operación segura"
        color_estado = "green"

    # --- GRÁFICOS ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(
        df_telemetria["indice_secuencial"],
        df_telemetria["vibracion_rms"],
        'gray', alpha=0.3, linewidth=0.8, label="Histórico completo"
    )
    ax1.plot(
        df_hist["indice_secuencial"],
        df_hist["vibracion_rms"],
        '#1f77b4', linewidth=2.5, label="Histórico hasta ahora"
    )
    if ciclos_rul < 999 and len(x_fut) > 0:
        ax1.plot(
            x_fut, y_proy,
            '#ff7f0e', linewidth=2, linestyle='--', label=f"Proyección ({modelo})"
        )
    ax1.axhline(
        y=UMBRAL_CRITICO, color='red', linestyle='--', linewidth=1.5,
        label=f"Umbral crítico ({UMBRAL_CRITICO:.2f} g)"
    )
    if ciclos_rul < 999 and horas_rul is not None:
        ax1.axvline(
            x=idx_actual + ciclos_rul, color='orange', linestyle=':',
            linewidth=2, label=f"RUL estimado: {formato_tiempo_legible(horas_rul)}"
        )
    ax1.set_xlabel("Ciclos", fontsize=10)
    ax1.set_ylabel("RMS (g)", fontsize=10)
    ax1.set_title("Evolución de la vibración", fontsize=11)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    if PARQUET_PATH.exists():
        df_bloque = pl.read_parquet(PARQUET_PATH).filter(pl.col("archivo_origen") == archivo_origen)
        senal = df_bloque["rodamiento_1"].head(PUNTOS_FFT).to_numpy()
        freqs, amps = ejecutar_fft(senal, FRECUENCIA_MUESTREO)
        mask = freqs <= 5000
        freqs = freqs[mask]
        amps = amps[mask]
        ax2.plot(freqs, amps, '#2ca02c', linewidth=1.2)
        ax2.axvline(x=freq_dom, color='red', linestyle='--', linewidth=1.5,
                    label=f"Dominante: {freq_dom:.1f} Hz")
        for f, label in [(236.4, "BPFO"), (296.8, "BPFI"), (139.2, "BSF"), (14.8, "FTF")]:
            ax2.axvline(x=f, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax2.set_xlabel("Frecuencia (Hz)", fontsize=10)
        ax2.set_ylabel("Amplitud (g)", fontsize=10)
        ax2.set_title("Espectro de Frecuencia", fontsize=11)
        ax2.set_xlim(0, 5000)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.2)
    else:
        ax2.text(0.5, 0.5, "Espectro no disponible\n(archivo Parquet ausente)",
                 ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    img_buffer.seek(0)

    # --- PDF ---
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        topMargin=0.8*inch,
        bottomMargin=0.8*inch,
        leftMargin=0.8*inch,
        rightMargin=0.8*inch
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
                                 fontSize=18, alignment=TA_CENTER, spaceAfter=12, textColor=colors.darkblue)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'],
                                   fontSize=13, spaceAfter=6, textColor=colors.darkblue)
    normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontSize=10, leading=14)
    center_style = ParagraphStyle('CenterStyle', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER)

    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ])

    story = []
    story.append(Paragraph("REPORTE DE PRONÓSTICO INDUSTRIAL", title_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"<b>Archivo analizado:</b> {archivo_origen}", normal_style))
    story.append(Paragraph(f"<b>Fecha de generación:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 0.15*inch))

    rul_text = formato_tiempo_legible(horas_rul) if horas_rul is not None else "Sin proyección"

    data = [
        ["Métrica", "Valor"],
        ["RMS Actual", f"{rms_actual:.4f} g"],
        ["RMS Máximo Histórico", f"{max_rms_historico:.4f} g"],
        ["Frecuencia Dominante", f"{freq_dom:.1f} Hz"],
        ["Amplitud Máxima", f"{amp_max:.4f} g"],
        ["RUL Estimado", rul_text],
        ["Modelo usado", modelo],
        ["Zona de Falla", zona_falla],
        ["Estado", estado],
    ]
    t = Table(data, colWidths=[2.5*inch, 3*inch])
    t.setStyle(table_style)
    story.append(t)
    story.append(Spacer(1, 0.2*inch))

    img = Image(img_buffer, width=7*inch, height=3.2*inch)
    story.append(img)

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>Recomendación</b>", heading_style))

    if max_rms_historico >= UMBRAL_CRITICO:
        rec = "🚨 <b>URGENTE:</b> El rodamiento ha alcanzado el umbral crítico. " \
              "Se requiere reemplazo inmediato antes de la próxima operación. " \
              "Riesgo de parada no planificada en las próximas horas."
        rec_color = colors.red
    elif max_rms_historico >= UMBRAL_ALERTA:
        if horas_rul is not None and horas_rul < 8:
            rec = f"⚠️ <b>ALERTA AVANZADA:</b> El desgaste es significativo. " \
                  f"Programar reemplazo en las próximas {formato_tiempo_legible(horas_rul)}. " \
                  "Revisar disponibilidad de repuestos."
            rec_color = colors.orange
        elif horas_rul is not None and horas_rul < 50:
            rec = f"🟡 <b>ALERTA INCIPIENTE:</b> Se detecta inicio de desgaste. " \
                  f"RUL estimado: {formato_tiempo_legible(horas_rul)}. " \
                  "Programar inspección visual y planificar reemplazo."
            rec_color = colors.gold
        else:
            rec = "🟣 <b>VIGILANCIA:</b> Se ha detectado daño pero la evolución es lenta. " \
                  "Mantener monitoreo periódico y registrar tendencia."
            rec_color = colors.purple
    else:
        rec = "✅ <b>OPERACIÓN NORMAL:</b> El rodamiento se encuentra dentro de " \
              "los parámetros operativos. Continuar con monitoreo de rutina."
        rec_color = colors.green

    rec_style = ParagraphStyle('RecStyle', parent=normal_style,
                               textColor=rec_color, fontSize=11, leading=15)
    story.append(Paragraph(rec, rec_style))

    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        "<i>Este reporte es generado automáticamente por el sistema de pronóstico industrial. "
        "Las decisiones de mantenimiento deben ser validadas por personal calificado.</i>",
        center_style
    ))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    archivo_prueba = "2004.02.12.10.32.39"
    try:
        path = generar_reporte_pdf(archivo_prueba)
        print(f"Reporte generado: {path}")
    except Exception as e:
        print(f"Error: {e}")
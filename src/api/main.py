"""
API REST para el sistema de pronóstico industrial.
Endpoints:
- GET /health: health check
- POST /predict: predicción de RUL y diagnóstico
- GET /report/{archivo_origen}: genera un reporte PDF
"""
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import tomllib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Añadir path para importar el core
sys.path.append(str(Path(__file__).parent.parent))
from core.dsp import calcular_rul_hibrido, ejecutar_fft, calcular_rms
from database.db_manager import DatabaseManager

# --- Configuración ---
CONFIG_PATH = Path("config.toml")
with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

UMBRAL_CRITICO = CONFIG["umbrales_severidad"]["critico_rms"]
UMBRAL_ALERTA = CONFIG["umbrales_severidad"]["alerta_rms"]
UMBRAL_APAGADO = CONFIG["umbrales_severidad"]["motor_apagado_rms"]
MINUTOS_POR_CICLO = CONFIG["diagnostico"]["minutos_por_ciclo"]
FRECUENCIA_MUESTREO = CONFIG["fisica_rodamiento"]["frecuencia_muestreo"]
PUNTOS_FFT = CONFIG["fisica_rodamiento"]["puntos_fft"]
TOLERANCIA_ARMONICA = CONFIG["diagnostico"]["tolerancia_armonica_hz"]

# --- Carga de datos (Lazy Loading para robustez en CI/CD) ---
TELEMETRIA_PATH = Path("datos/telemetria_optimizacion.csv")
PARQUET_PATH = Path("datos/nasa_bearing_consolidado.parquet")

df_telemetria = None
df_parquet = None

def inicializar_datos():
    global df_telemetria, df_parquet
    if df_telemetria is None and TELEMETRIA_PATH.exists():
        df_telemetria = pl.read_csv(TELEMETRIA_PATH)
    if df_parquet is None and PARQUET_PATH.exists():
        df_parquet = pl.read_parquet(PARQUET_PATH)

# Frecuencias teóricas
FREQ_BPFO = 236.4
FREQ_BPFI = 296.8
FREQ_BSF = 139.2
FREQ_FTF = 14.8

# --- Modelos de respuesta ---
class PredictResponse(BaseModel):
    archivo_origen: str
    rms_actual: float
    max_rms_historico: float
    rul_hours: float
    rul_ci_lower: Optional[float] = None
    rul_ci_upper: Optional[float] = None
    frecuencia_dominante: float
    modelo_usado: str
    zona_falla: str
    estado: str

class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.1"
    telemetria_registros: int


# --- Helpers ---
def obtener_intervalo_confianza(x, y, umbral, n_bootstrap=100):
    n = len(x)
    if n < 10:
        return None, None
    rul_estimados = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        x_boot = x[idx]
        y_boot = y[idx]
        try:
            ciclos, _, _, _ = calcular_rul_hibrido(x_boot, y_boot, umbral)
            if ciclos < 999:
                rul_estimados.append(ciclos)
        except:
            continue
    if len(rul_estimados) < 20:
        return None, None
    ci_lower = np.percentile(rul_estimados, 5)
    ci_upper = np.percentile(rul_estimados, 95)
    return ci_lower, ci_upper


def verificar_armonicos(freq_dom: float, freq_base: float, max_orden: int = 5) -> bool:
    for orden in range(1, max_orden + 1):
        if abs(freq_dom - (freq_base * orden)) <= TOLERANCIA_ARMONICA:
            return True
    return False


def determinar_zona_falla(freq_dom: float, max_rms: float) -> str:
    if max_rms < UMBRAL_ALERTA:
        return "Ninguna (operación normal)"
    if verificar_armonicos(freq_dom, FREQ_BPFO) or freq_dom > 3500:
        return "PISTA EXTERNA (BPFO)"
    if verificar_armonicos(freq_dom, FREQ_BPFI):
        return "PISTA INTERNA (BPFI)"
    if verificar_armonicos(freq_dom, FREQ_BSF):
        return "ELEMENTOS RODANTES (BSF)"
    if verificar_armonicos(freq_dom, FREQ_FTF):
        return "JAULA (FTF)"
    return "PISTA EXTERNA (Modulación)"


def calcular_estado(max_rms: float, ciclos_rul: float) -> str:
    if max_rms >= UMBRAL_CRITICO:
        return "CRÍTICO (Reemplazo inmediato)"
    elif max_rms >= UMBRAL_ALERTA:
        if ciclos_rul < 50:
            return "ALERTA AVANZADA"
        elif ciclos_rul < 200:
            return "ALERTA INCIPIENTE"
        else:
            return "VIGILANCIA"
    else:
        return "SALUDABLE"


# --- FastAPI App ---
app = FastAPI(
    title="Industrial AI Prognostics API",
    description="API para pronóstico de RUL y diagnóstico de rodamientos",
    version="2.0.1"
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    inicializar_datos()
    return HealthResponse(
        status="ok",
        telemetria_registros=df_telemetria.height if df_telemetria is not None else 0
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    archivo_origen: str = Query(..., description="Nombre del archivo de origen (ej. 2004.02.18.20.42.39)")
):
    inicializar_datos()
    if df_telemetria is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio no disponible: La base de datos de telemetría no ha sido cargada o el pipeline no se ha ejecutado."
        )
    df_fila = df_telemetria.filter(pl.col("archivo_origen") == archivo_origen)
    if df_fila.is_empty():
        raise HTTPException(status_code=404, detail=f"Archivo {archivo_origen} no encontrado")

    datos = df_fila.to_dicts()[0]
    idx_actual = datos["indice_secuencial"]
    rms_actual = datos["vibracion_rms"]
    freq_dom = datos["frecuencia_dominante_hz"]

    df_hist = df_telemetria.filter(pl.col("indice_secuencial") <= idx_actual)
    max_rms_historico = df_hist["vibracion_rms"].max()

    x_ciclos = df_hist["indice_secuencial"].to_numpy()
    y_rms = df_hist["vibracion_rms"].to_numpy()

    ciclos_rul, _, _, modelo = calcular_rul_hibrido(x_ciclos, y_rms, UMBRAL_CRITICO)
    horas_rul = (ciclos_rul * MINUTOS_POR_CICLO) / 60.0

    ci_lower, ci_upper = obtener_intervalo_confianza(x_ciclos, y_rms, UMBRAL_CRITICO)
    if ci_lower is not None:
        ci_lower_hours = (ci_lower * MINUTOS_POR_CICLO) / 60.0
        ci_upper_hours = (ci_upper * MINUTOS_POR_CICLO) / 60.0
    else:
        ci_lower_hours = ci_upper_hours = None

    zona = determinar_zona_falla(freq_dom, max_rms_historico)
    estado = calcular_estado(max_rms_historico, ciclos_rul)

    return PredictResponse(
        archivo_origen=archivo_origen,
        rms_actual=round(rms_actual, 4),
        max_rms_historico=round(max_rms_historico, 4),
        rul_hours=round(horas_rul, 1),
        rul_ci_lower=round(ci_lower_hours, 1) if ci_lower_hours else None,
        rul_ci_upper=round(ci_upper_hours, 1) if ci_upper_hours else None,
        frecuencia_dominante=round(freq_dom, 1),
        modelo_usado=modelo,
        zona_falla=zona,
        estado=estado
    )


@app.get("/report/{archivo_origen}")
async def generate_report(archivo_origen: str):
    try:
        from ..report.pdf_generator import generar_reporte_pdf
        pdf_path = generar_reporte_pdf(archivo_origen)
        return {"message": "Reporte generado", "path": str(pdf_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
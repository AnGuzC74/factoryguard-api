"""
API REST para el sistema de pronóstico industrial.
Endpoints:
- GET /health: health check
- POST /predict: predicción de RUL y diagnóstico
- GET /report/{archivo_origen}: genera un reporte PDF
- POST /predict/classify: clasifica fallas según características operativas
- POST /agent/trigger: dispara el agente prescriptivo
- GET /agent/status/{agent_run_id}: estado de la sesión del agente
- POST /agent/approve/{agent_run_id}: aprueba o rechaza la orden
- POST /agent/ask/{agent_run_id}: pregunta en lenguaje natural sobre la sesión
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import polars as pl
import tomllib
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Añadir path para importar el core
sys.path.append(str(Path(__file__).parent.parent))
from core.dsp import calcular_rul_hibrido, ejecutar_fft, calcular_rms, demodular_envolvente
from database.db_manager import DatabaseManager
from services.fault_classifier import ClasificadorFalla
from agent_orchestrator.graph import OrquestadorAgentePrescriptivo
from agent.rag_agent import RAGAgent

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

AGENT_CONFIG = CONFIG.get("agent", {})
FEATURE_FLAG_ENABLED = AGENT_CONFIG.get("feature_flag_enabled", False)

# --- Base de Datos ---
DB_PATH = CONFIG["infraestructura"].get("database", "datos/industrial_ai.db")
db_manager = DatabaseManager(DB_PATH)

# --- Inicializar Clasificador ---
classifier = ClasificadorFalla()
classifier_loaded = False

def cargar_clasificador():
    global classifier, classifier_loaded
    if not classifier_loaded:
        model_path = Path("datos/fault_classifier.pkl")
        if model_path.exists():
            classifier.cargar_modelo(str(model_path))
            classifier_loaded = True
            print("[API] Clasificador de fallas cargado correctamente.")
        else:
            print("[API] Advertencia: No se encontró el clasificador de fallas entrenado en datos/fault_classifier.pkl")

# --- Inicializar Orquestador ---
orquestador = OrquestadorAgentePrescriptivo(db_manager)

# --- Inicializar RAGAgent ---
rag_agent = RAGAgent(CONFIG_PATH)

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
    espectro_envolvente_frecuencias: Optional[list[float]] = None
    espectro_envolvente_amplitudes: Optional[list[float]] = None
    tipo_falla_predicho: Optional[str] = None
    confianza: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.1"
    telemetria_registros: int

class ClassifyRequest(BaseModel):
    type_product: str  # "L", "M", "H"
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

class ClassifyResponse(BaseModel):
    tipo_falla_predicho: str
    confianza: float

class AgentTriggerRequest(BaseModel):
    asset_id: int
    rul_hours: float
    tipo_falla: str
    severidad: str

class AgentApproveRequest(BaseModel):
    aprobado: bool

class AgentAskRequest(BaseModel):
    pregunta: str


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


def verificar_feature_flag():
    if not FEATURE_FLAG_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Servicio no disponible: El agente prescriptivo está desactivado en la configuración."
        )


# --- FastAPI App ---
app = FastAPI(
    title="Industrial AI Prognostics API",
    description="API para pronóstico de RUL, diagnóstico y orquestación prescriptiva",
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

    # 1. Demodular envolvente si df_parquet y la columna están disponibles
    frecuencias_env_list = []
    amplitudes_env_list = []
    if df_parquet is not None and "rodamiento_1" in df_parquet.columns:
        df_bloque = df_parquet.filter(pl.col("archivo_origen") == archivo_origen)
        if not df_bloque.is_empty():
            senal_cruda = df_bloque["rodamiento_1"].head(PUNTOS_FFT).to_numpy()
            f_env, a_env = demodular_envolvente(senal_cruda, FRECUENCIA_MUESTREO)
            frecuencias_env_list = f_env.tolist()
            amplitudes_env_list = a_env.tolist()

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
        estado=estado,
        espectro_envolvente_frecuencias=[round(f, 2) for f in frecuencias_env_list] if frecuencias_env_list else None,
        espectro_envolvente_amplitudes=[round(a, 6) for a in amplitudes_env_list] if amplitudes_env_list else None,
        tipo_falla_predicho=None,
        confianza=None
    )


@app.get("/report/{archivo_origen}")
async def generate_report(archivo_origen: str):
    try:
        from ..report.pdf_generator import generar_reporte_pdf
        pdf_path = generar_reporte_pdf(archivo_origen)
        return {"message": "Reporte generado", "path": str(pdf_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Nuevos Endpoints (Clasificación de fallas y Agente Prescriptivo) ---

@app.post("/predict/classify", response_model=ClassifyResponse)
async def predict_classify(request: ClassifyRequest):
    cargar_clasificador()
    if not classifier.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Servicio no disponible: El modelo de clasificación no ha sido entrenado."
        )

    features_dict = {
        "Type": request.type_product,
        "Air temperature [K]": request.air_temperature_k,
        "Process temperature [K]": request.process_temperature_k,
        "Rotational speed [rpm]": request.rotational_speed_rpm,
        "Torque [Nm]": request.torque_nm,
        "Tool wear [min]": request.tool_wear_min
    }

    try:
        tipo_falla, confianza = classifier.predecir(features_dict)
        return ClassifyResponse(
            tipo_falla_predicho=tipo_falla,
            confianza=confianza
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


@app.post("/agent/trigger")
async def agent_trigger(request: AgentTriggerRequest):
    verificar_feature_flag()
    try:
        session_id = orquestador.disparar_grafo(
            asset_id=request.asset_id,
            rul_hours=request.rul_hours,
            tipo_falla=request.tipo_falla,
            severidad=request.severidad
        )
        return {
            "agent_run_id": session_id,
            "status": "Pausado (Esperando Aprobación)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando el agente: {str(e)}")


@app.get("/agent/status/{agent_run_id}")
async def agent_status(agent_run_id: str):
    verificar_feature_flag()
    session = db_manager.get_agent_session(agent_run_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Sesión del agente {agent_run_id} no encontrada.")
    return {
        "agent_run_id": session["session_id"],
        "status": session["status"],
        "state_name": session["state_name"],
        "state_data": session["state_data"]
    }


@app.post("/agent/approve/{agent_run_id}")
async def agent_approve(agent_run_id: str, request: AgentApproveRequest):
    verificar_feature_flag()
    session = db_manager.get_agent_session(agent_run_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Sesión {agent_run_id} no encontrada.")

    try:
        final_state = orquestador.procesar_aprobacion(agent_run_id, request.aprobado)
        if not final_state:
            raise HTTPException(status_code=500, detail="Error procesando la aprobación del agente.")
        return {
            "status": "Aprobado" if request.aprobado else "Rechazado",
            "mensaje_final": final_state["mensaje_final"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la aprobación: {str(e)}")


@app.post("/agent/ask/{agent_run_id}")
async def agent_ask(agent_run_id: str, request: AgentAskRequest):
    verificar_feature_flag()
    session = db_manager.get_agent_session(agent_run_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Sesión del agente {agent_run_id} no encontrada.")

    state_data = session["state_data"]

    # Enriquecer el prompt con la información del estado actual de la sesión
    orden = state_data.get("orden_prescriptiva") or {}
    recomendacion = state_data.get("recomendacion") or {}
    best_balance = recomendacion.get("mejor_balance") or {}

    contexto = (
        f"Contexto del diagnóstico actual:\n"
        f"- ID de Activo: {state_data.get('asset_id')}\n"
        f"- Tipo de Falla: {state_data.get('tipo_falla')}\n"
        f"- Severidad: {state_data.get('severidad')}\n"
        f"- Vida Útil Restante (RUL): {state_data.get('rul_hours')} horas\n"
        f"- Urgencia: {orden.get('urgencia', 'BAJA')}\n"
        f"- Acción Sugerida: {orden.get('accion_sugerida')}\n"
        f"- Requiere Reemplazo: {'Sí' if orden.get('requiere_reemplazo') else 'No'}\n"
    )
    if best_balance:
        contexto += (
            f"- Proveedor Recomendado: {best_balance.get('proveedor')}\n"
            f"- Precio: {best_balance.get('precio')} EUR\n"
            f"- Tiempo de arribo: {best_balance.get('tiempo_arribo_dias')} días\n"
        )
    contexto += f"- Decisión de Operador: {state_data.get('aprobado')}\n"
    contexto += f"- Mensaje de Cierre: {state_data.get('mensaje_final')}\n"

    full_query = (
        f"Usa la siguiente información del estado del agente para responder la pregunta del usuario:\n"
        f"=== ESTADO ===\n{contexto}\n=============\n"
        f"Pregunta del usuario: {request.pregunta}"
    )

    try:
        respuesta = rag_agent.responder_conversacional(full_query)
        return {"respuesta": respuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando al RAGAgent: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

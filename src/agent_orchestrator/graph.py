"""
Orquestador de Agente Prescriptivo basado en LangGraph.

Este módulo define la arquitectura de StateGraph para la toma de decisiones determinista,
la búsqueda de repuestos y la gestión de aprobaciones humanas.
"""

from typing import TypedDict, List, Dict, Any, Optional, Tuple
import uuid
import json
from langgraph.graph import StateGraph, END

from src.adapters.mock_parts_supplier import MockPartsSupplier
from src.database.db_manager import DatabaseManager


# Constantes Nombradas - Reglas deterministas para generar la orden prescriptiva
# Formato de regla por severidad
REGLAS_PRESCRIPTIVAS = {
    "CRÍTICO": {
        "urgencia": "CRÍTICA",
        "requiere_reemplazo": True,
        "accion": "Detener el equipo inmediatamente, aplicar protocolo LOTO, desmontar rodamiento y reemplazar por una nueva unidad. Inspeccionar el eje para descartar daños colaterales."
    },
    "ALERTA AVANZADA": {
        "urgencia": "ALTA",
        "requiere_reemplazo": True,
        "accion": "Programar el reemplazo del rodamiento dentro de las próximas 24-48 horas. Aumentar la frecuencia de monitoreo térmico y vibratorio en el interín."
    },
    "ALERTA INCIPIENTE": {
        "urgencia": "MEDIA",
        "requiere_reemplazo": False,
        "accion": "Inspeccionar visualmente el rodamiento, verificar el nivel y calidad de la lubricación. Re-engrasar si es necesario y registrar evolución en 12 horas."
    },
    "VIGILANCIA": {
        "urgencia": "BAJA",
        "requiere_reemplazo": False,
        "accion": "Continuar con el monitoreo rutinario. Verificar alineación láser del acoplamiento y asegurar que no existan cargas anómalas en el eje."
    },
    "SALUDABLE": {
        "urgencia": "BAJA",
        "requiere_reemplazo": False,
        "accion": "No se requieren acciones correctivas. Mantener el plan de mantenimiento preventivo estándar."
    }
}


def normalizar_severidad(sev_str: str) -> str:
    """
    Normaliza la severidad del diagnóstico para mapearla con las constantes deterministas.
    """
    if not sev_str:
        return "SALUDABLE"
    sev_upper = sev_str.upper()
    if "CRÍTICO" in sev_upper or "CRITICO" in sev_upper:
        return "CRÍTICO"
    if "ALERTA AVANZADA" in sev_upper:
        return "ALERTA AVANZADA"
    if "ALERTA INCIPIENTE" in sev_upper:
        return "ALERTA INCIPIENTE"
    if "VIGILANCIA" in sev_upper:
        return "VIGILANCIA"
    return "SALUDABLE"


class AgentState(TypedDict):
    """
    Esquema de estado para la sesión de orquestación del agente.
    """
    session_id: str
    asset_id: int
    rul_hours: float
    tipo_falla: str
    severidad: str
    orden_prescriptiva: Optional[Dict[str, Any]]
    repuestos: List[Dict[str, Any]]
    tabla_comparativa: List[Dict[str, Any]]
    recomendacion: Optional[Dict[str, Any]]
    aprobado: Optional[str]  # "PENDIENTE", "APROBADO", "RECHAZADO"
    mensaje_final: Optional[str]
    reporte_juez: Optional[Dict[str, Any]]
    debate: Optional[Dict[str, Any]]


# --- Nodos del Grafo ---

def diagnosticar(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que consume el diagnóstico de la fase 1 (RUL y clasificación de falla).
    """
    # Solo consume y registra, no altera nada.
    return {
        "session_id": state["session_id"],
        "asset_id": state["asset_id"],
        "rul_hours": state["rul_hours"],
        "tipo_falla": state["tipo_falla"],
        "severidad": state["severidad"]
    }


def generar_orden_prescriptiva(state: AgentState) -> Dict[str, Any]:
    """
    Genera de forma determinista la orden de inspección y reemplazo según reglas fijas.
    """
    sev_norm = normalizar_severidad(state.get("severidad", "SALUDABLE"))
    regla = REGLAS_PRESCRIPTIVAS.get(sev_norm, REGLAS_PRESCRIPTIVAS["SALUDABLE"])

    orden = {
        "urgencia": regla["urgencia"],
        "requiere_reemplazo": regla["requiere_reemplazo"],
        "accion_sugerida": regla["accion"],
        "inspeccion_requerida": f"Verificación del componente de rodamiento asociado a {state.get('tipo_falla', 'Falla General')}"
    }
    return {"orden_prescriptiva": orden}


def buscar_repuestos(state: AgentState) -> Dict[str, Any]:
    """
    Invocación determinista al mock de proveedores de repuestos si se requiere reemplazo.
    """
    orden = state.get("orden_prescriptiva") or {}
    requiere_reemplazo = orden.get("requiere_reemplazo", False)

    repuestos = []
    if requiere_reemplazo:
        supplier = MockPartsSupplier()
        repuestos = supplier.buscar_repuestos(state.get("tipo_falla", "DEFAULT"))

    return {"repuestos": repuestos}


def negociacion_multiagente(state: AgentState) -> Dict[str, Any]:
    """
    Nodo de debate multi-agente donde Operaciones, Logística y Finanzas discuten
    el plan propuesto y el Juez emite una evaluación determinista.
    """
    from src.agent_orchestrator.negotiation import ejecutar_debate_y_evaluacion
    db = DatabaseManager()

    # Obtener criticidad del activo de la base de datos
    criticidad_db = None
    asset_id = state.get("asset_id")
    if asset_id:
        asset_data = db.get_asset_by_id(asset_id)
        if asset_data:
            criticidad_db = asset_data.get("criticidad")

    # Ejecutar debate y evaluación continua del Juez
    reporte, debate_payload = ejecutar_debate_y_evaluacion(
        asset_id=state.get("asset_id", 1),
        rul_hours=state.get("rul_hours", 100.0),
        tipo_falla=state.get("tipo_falla", "DEFAULT"),
        severidad=state.get("severidad", "SALUDABLE"),
        all_repuestos=state.get("repuestos", []),
        recomendacion_balance=state.get("recomendacion", {}),
        criticidad_db=criticidad_db
    )

    return {
        "reporte_juez": reporte.model_dump(),
        "debate": debate_payload
    }


def presentar_comparativa(state: AgentState) -> Dict[str, Any]:
    """
    Arma una tabla comparativa y determina la recomendación en base a reglas deterministas.
    """
    repuestos = state.get("repuestos") or []
    tabla_comparativa = []
    recomendacion = None

    if repuestos:
        mejor_precio = min(repuestos, key=lambda x: x["precio"])
        mas_rapido = min(repuestos, key=lambda x: x["tiempo_arribo_dias"])

        # Fórmula de balance determinista: Peso 50% precio, 50% tiempo de entrega
        # Puntuación menor es mejor.
        mejor_balance = min(
            repuestos,
            key=lambda x: (x["precio"] / 1000.0) * 0.5 + (x["tiempo_arribo_dias"] / 10.0) * 0.5
        )

        tabla_comparativa = repuestos
        recomendacion = {
            "mejor_precio": mejor_precio,
            "mas_rapido": mas_rapido,
            "mejor_balance": mejor_balance,
            "justificacion_balance": (
                f"Se recomienda el proveedor {mejor_balance['proveedor']} por ofrecer la mejor "
                f"relación precio-tiempo ({mejor_balance['precio']} EUR, {mejor_balance['tiempo_arribo_dias']} días)."
            )
        }

    return {
        "tabla_comparativa": tabla_comparativa,
        "recomendacion": recomendacion
    }


def aprobacion_human_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo de pausa que marca el estado como PENDIENTE de aprobación externa.
    """
    return {"aprobado": "PENDIENTE"}


def finalizar_orden(state: AgentState) -> Dict[str, Any]:
    """
    Nodo final que concluye la orden en base a la decisión humana de aprobación.
    """
    decision = state.get("aprobado", "PENDIENTE")
    recomendacion = state.get("recomendacion")

    if decision == "APROBADO":
        if recomendacion and recomendacion.get("mejor_balance"):
            prov = recomendacion["mejor_balance"]["proveedor"]
            precio = recomendacion["mejor_balance"]["precio"]
            msg = (
                f"Orden de compra confirmada con el proveedor {prov} por un monto de {precio} EUR. "
                "La orden ha sido enviada al departamento de compras y el despliegue de mantenimiento está programado."
            )
        else:
            msg = "Orden de inspección y mantenimiento preventivo confirmada por el operador."
    elif decision == "RECHAZADO":
        msg = "La propuesta de orden de compra/inspección ha sido expresamente RECHAZADA por el operador. Operación cancelada."
    else:
        msg = "La orden se encuentra en espera de definición por el operador."

    return {"mensaje_final": msg}


# --- Configuración y Construcción del StateGraph ---

def construir_grafo() -> StateGraph:
    """
    Construye y configura el StateGraph del agente prescriptivo.
    """
    workflow = StateGraph(AgentState)

    # Añadir nodos
    workflow.add_node("diagnosticar", diagnosticar)
    workflow.add_node("generar_orden_prescriptiva", generar_orden_prescriptiva)
    workflow.add_node("buscar_repuestos", buscar_repuestos)
    workflow.add_node("presentar_comparativa", presentar_comparativa)
    workflow.add_node("negociacion_multiagente", negociacion_multiagente)
    workflow.add_node("aprobacion_humana", aprobacion_human_node)
    workflow.add_node("finalizar_orden", finalizar_orden)

    # Configurar bordes
    workflow.set_entry_point("diagnosticar")
    workflow.add_edge("diagnosticar", "generar_orden_prescriptiva")
    workflow.add_edge("generar_orden_prescriptiva", "buscar_repuestos")
    workflow.add_edge("buscar_repuestos", "presentar_comparativa")
    workflow.add_edge("presentar_comparativa", "negociacion_multiagente")
    workflow.add_edge("negociacion_multiagente", "aprobacion_humana")
    workflow.add_edge("aprobacion_humana", "finalizar_orden")
    workflow.add_edge("finalizar_orden", END)

    return workflow


class OrquestadorAgentePrescriptivo:
    """
    Orquestador de alto nivel para gestionar la inicialización, pausa y reanudación
    del grafo de estados del agente prescriptivo utilizando SQLite como persistencia de estado.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.grafo = construir_grafo().compile()

    def disparar_grafo(self, asset_id: int, rul_hours: float, tipo_falla: str, severidad: str) -> str:
        """
        Inicia la ejecución del grafo hasta llegar al nodo de aprobación humana.
        """
        session_id = str(uuid.uuid4())

        # Estado inicial
        state: AgentState = {
            "session_id": session_id,
            "asset_id": asset_id,
            "rul_hours": rul_hours,
            "tipo_falla": tipo_falla,
            "severidad": severidad,
            "orden_prescriptiva": None,
            "repuestos": [],
            "tabla_comparativa": [],
            "recomendacion": None,
            "aprobado": "PENDIENTE",
            "mensaje_final": None,
            "reporte_juez": None,
            "debate": None
        }

        # Ejecutar los primeros nodos del grafo secuencialmente
        # En LangGraph, podemos simular la ejecución paso a paso o invocar nodos directamente.
        # Dado que queremos pausar deterministamente en 'aprobacion_humana' y persistir,
        # ejecutamos la cadena de transformaciones hasta el nodo de aprobación:
        state = {**state, **diagnosticar(state)}
        state = {**state, **generar_orden_prescriptiva(state)}
        state = {**state, **buscar_repuestos(state)}
        state = {**state, **presentar_comparativa(state)}
        state = {**state, **negociacion_multiagente(state)}
        state = {**state, **aprobacion_human_node(state)}

        # Determinar estatus según la evaluación del Juez
        status = "Pausado (Esperando Aprobación)"
        if state.get("reporte_juez") and not state["reporte_juez"]["evaluacion_global"]["aprobado"]:
            status = "Pausado (Rechazado por Juez)"

        # Persistir estado intermedio en SQLite
        self.db.save_agent_session(
            session_id=session_id,
            state_name="aprobacion_humana",
            state_data=state,
            status=status
        )

        return session_id

    def procesar_aprobacion(self, session_id: str, aprobado: bool) -> Optional[Dict[str, Any]]:
        """
        Reanuda la ejecución del grafo desde el nodo de aprobación humana con la decisión dada.
        """
        session = self.db.get_agent_session(session_id)
        if not session:
            return None

        state = session["state_data"]

        # Actualizar la decisión humana
        state["aprobado"] = "APROBADO" if aprobado else "RECHAZADO"

        # Ejecutar el nodo de finalización de orden
        res_final = finalizar_orden(state)
        state = {**state, **res_final}

        # Guardar estado final en SQLite
        self.db.save_agent_session(
            session_id=session_id,
            state_name="finalizado",
            state_data=state,
            status="Aprobado" if aprobado else "Rechazado"
        )

        return state

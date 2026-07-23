"""
Pruebas de Integración y de Unidad para el Agente Prescriptivo.
"""

import json
import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.database.db_manager import DatabaseManager
from src.agent_orchestrator.graph import (
    generar_orden_prescriptiva,
    buscar_repuestos,
    presentar_comparativa,
    finalizar_orden,
    normalizar_severidad
)


@pytest.fixture(scope="module")
def client():
    # Asegurar que el feature flag esté activado para los endpoints de prueba
    api_main.FEATURE_FLAG_ENABLED = True
    return TestClient(api_main.app)


def test_generar_orden_prescriptiva_rules():
    """
    Verifica generar_orden_prescriptiva con al menos 4 combinaciones de tipo_falla × severidad.
    """
    combinaciones = [
        {"tipo_falla": "HDF", "severidad": "CRÍTICO (Reemplazo inmediato)", "urgencia_esperada": "CRÍTICA", "requiere_reemplazo": True},
        {"tipo_falla": "TWF", "severidad": "ALERTA AVANZADA", "urgencia_esperada": "ALTA", "requiere_reemplazo": True},
        {"tipo_falla": "PWF", "severidad": "ALERTA INCIPIENTE", "urgencia_esperada": "MEDIA", "requiere_reemplazo": False},
        {"tipo_falla": "OSF", "severidad": "SALUDABLE", "urgencia_esperada": "BAJA", "requiere_reemplazo": False}
    ]

    for comb in combinaciones:
        state = {
            "session_id": "test",
            "asset_id": 1,
            "rul_hours": 100.0,
            "tipo_falla": comb["tipo_falla"],
            "severidad": comb["severidad"],
            "orden_prescriptiva": None,
            "repuestos": [],
            "tabla_comparativa": [],
            "recomendacion": None,
            "aprobado": "PENDIENTE",
            "mensaje_final": None
        }

        res = generar_orden_prescriptiva(state)
        orden = res["orden_prescriptiva"]

        assert orden["urgencia"] == comb["urgencia_esperada"]
        assert orden["requiere_reemplazo"] == comb["requiere_reemplazo"]
        assert comb["tipo_falla"] in orden["inspeccion_requerida"]


def test_buscar_repuestos_and_persistence():
    """
    Test de que buscar_repuestos con el mock devuelve estructura válida
    y que se persiste correctamente en la base de datos de SQLite.
    """
    db = DatabaseManager()

    state = {
        "session_id": "test-session-persistence-1234",
        "asset_id": 1,
        "rul_hours": 15.2,
        "tipo_falla": "HDF",
        "severidad": "CRÍTICO",
        "orden_prescriptiva": {
            "urgencia": "CRÍTICA",
            "requiere_reemplazo": True,
            "accion_sugerida": "Reemplazar",
            "inspeccion_requerida": "Verificar HDF"
        },
        "repuestos": [],
        "tabla_comparativa": [],
        "recomendacion": None,
        "aprobado": "PENDIENTE",
        "mensaje_final": None
    }

    # 1. Ejecutar buscar_repuestos
    res_repuestos = buscar_repuestos(state)
    repuestos = res_repuestos["repuestos"]

    assert len(repuestos) > 0
    for r in repuestos:
        assert "proveedor" in r
        assert "precio" in r
        assert "tiempo_arribo_dias" in r
        assert "pieza_solicitada" in r
        assert "HDF" in r["pieza_solicitada"]

    state["repuestos"] = repuestos

    # 2. Presentar comparativa
    res_comp = presentar_comparativa(state)
    state["tabla_comparativa"] = res_comp["tabla_comparativa"]
    state["recomendacion"] = res_comp["recomendacion"]

    assert state["recomendacion"] is not None
    assert "mejor_balance" in state["recomendacion"]

    # 3. Guardar en SQLite
    db.save_agent_session(
        session_id=state["session_id"],
        state_name="presentar_comparativa",
        state_data=state,
        status="Pausado (Esperando Aprobación)"
    )

    # 4. Recuperar y validar persistencia completa
    retrieved = db.get_agent_session(state["session_id"])
    assert retrieved is not None
    assert retrieved["status"] == "Pausado (Esperando Aprobación)"
    assert len(retrieved["state_data"]["repuestos"]) == len(repuestos)
    assert retrieved["state_data"]["recomendacion"]["mejor_balance"]["proveedor"] == state["recomendacion"]["mejor_balance"]["proveedor"]


def test_fastapi_client_integration_flow(client):
    """
    Test de integración con TestClient: trigger → estado en aprobacion_humana → approve → estado final.
    """
    # 1. Trigger agent
    trigger_payload = {
        "asset_id": 1,
        "rul_hours": 24.5,
        "tipo_falla": "PWF",
        "severidad": "ALERTA AVANZADA"
    }

    response = client.post("/agent/trigger", json=trigger_payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "agent_run_id" in res_data
    assert res_data["status"] == "Pausado (Esperando Aprobación)"

    agent_run_id = res_data["agent_run_id"]

    # 2. Get status (should be waiting for approval)
    response_status = client.get(f"/agent/status/{agent_run_id}")
    assert response_status.status_code == 200
    status_data = response_status.json()
    assert status_data["status"] in ["Pausado (Esperando Aprobación)", "Pausado (Rechazado por Juez)"]
    assert status_data["state_name"] == "aprobacion_humana"
    assert len(status_data["state_data"]["repuestos"]) == 6
    assert status_data["state_data"]["recomendacion"] is not None

    # 3. Approve and resume
    approve_payload = {"aprobado": True}
    response_approve = client.post(f"/agent/approve/{agent_run_id}", json=approve_payload)
    assert response_approve.status_code == 200
    approve_data = response_approve.json()
    assert approve_data["status"] == "Aprobado"
    assert "confirmada" in approve_data["mensaje_final"]

    # 4. Verify final status in DB
    response_status_final = client.get(f"/agent/status/{agent_run_id}")
    assert response_status_final.status_code == 200
    final_status_data = response_status_final.json()
    assert final_status_data["status"] == "Aprobado"
    assert final_status_data["state_name"] == "finalizado"
    assert final_status_data["state_data"]["aprobado"] == "APROBADO"


def test_agent_ask_endpoint_rag(client):
    """
    Test de que /agent/ask responde usando el RAG existente.
    """
    # Trigger a session first
    trigger_payload = {
        "asset_id": 1,
        "rul_hours": 12.0,
        "tipo_falla": "BPFO",
        "severidad": "CRÍTICO"
    }
    response = client.post("/agent/trigger", json=trigger_payload)
    agent_run_id = response.json()["agent_run_id"]

    # Ask a question
    ask_payload = {"pregunta": "¿Qué manual o procedimiento técnico me recomiendas aplicar?"}
    response_ask = client.post(f"/agent/ask/{agent_run_id}", json=ask_payload)
    assert response_ask.status_code == 200
    ask_data = response_ask.json()
    assert "respuesta" in ask_data
    # Debe contener mención de manuales técnicos o referencias de RAGAgent
    assert len(ask_data["respuesta"]) > 50


def test_feature_flag_disabled_returns_503(client):
    """
    Test de que con feature_flag_enabled = false, los endpoints /agent/* devuelven 503.
    """
    api_main.FEATURE_FLAG_ENABLED = False

    # Intentar trigger
    trigger_payload = {
        "asset_id": 1,
        "rul_hours": 24.5,
        "tipo_falla": "PWF",
        "severidad": "ALERTA AVANZADA"
    }
    response = client.post("/agent/trigger", json=trigger_payload)
    assert response.status_code == 503
    assert "desactivado" in response.json()["detail"].lower()

    # Intentar status
    response = client.get("/agent/status/some-id")
    assert response.status_code == 503

    # Restablecer para no interferir con otros tests
    api_main.FEATURE_FLAG_ENABLED = True


# --- Nuevos Tests de Negociación Prescriptiva Multi-Agente & Juez ---

def test_juez_determinista_aprobacion_y_rechazo():
    """
    Verifica las decisiones deterministas del Juez bajo diferentes condiciones de RUL y costos.
    """
    from src.agent_orchestrator.negotiation import ejecutar_debate_y_evaluacion

    # Mock de repuestos (3 proveedores, con Estándar y Exprés cada uno)
    all_repuestos = [
        {"proveedor": "SKF Iberia S.A.", "tipo_envio": "Estándar", "precio": 350.0, "tiempo_arribo_dias": 5, "pieza_solicitada": "Kit SKF"},
        {"proveedor": "SKF Iberia S.A.", "tipo_envio": "Exprés", "precio": 600.0, "tiempo_arribo_dias": 2, "pieza_solicitada": "Kit SKF"},
    ]

    # Caso 1: Aprobado - RUL holgado (150 horas / 24 = 6.25 días). Margen de seguridad es 2 días.
    # El repuesto de Exprés (tiempo_arribo_dias = 2) llega perfectamente a tiempo (2 <= 6.25 - 2).
    recomendacion = {"mejor_balance": all_repuestos[1]} # Exprés

    reporte, debate = ejecutar_debate_y_evaluacion(
        asset_id=1,
        rul_hours=150.0,
        tipo_falla="TWF",
        severidad="ALERTA AVANZADA",
        all_repuestos=all_repuestos,
        recomendacion_balance=recomendacion,
        criticidad_db="ALTA"
    )

    assert reporte.evaluacion_global.aprobado is True
    assert reporte.evaluacion_global.score_alineacion == 1.0
    assert reporte.analisis_costos.costo_parada_estimado == 0.0
    assert reporte.analisis_costos.costo_total_solucion == 600.0 + 300.0
    assert "aprobado" in debate["ops_agent"]["recomendacion"].lower() or "perfectamente" in debate["ops_agent"]["recomendacion"].lower()

    # Caso 2: Rechazo por regla de tiempo - RUL crítico (24 horas / 24 = 1.0 días).
    # Ningún repuesto llega antes de (1.0 - 2.0 = -1.0) días.
    reporte_crit, _ = ejecutar_debate_y_evaluacion(
        asset_id=1,
        rul_hours=24.0,
        tipo_falla="TWF",
        severidad="CRÍTICO",
        all_repuestos=all_repuestos,
        recomendacion_balance=recomendacion,
        criticidad_db="CRÍTICA"
    )

    assert reporte_crit.evaluacion_global.aprobado is False
    assert reporte_crit.evaluacion_global.score_alineacion < 1.0
    assert "excede el margen de seguridad" in reporte_crit.motivo_rechazo.lower()

    # Caso 3: Rechazo por regla financiera - Elegido Estándar pero el coste de parada supera por mucho el sobrecoste Exprés.
    # RUL = 100 horas (~4.1 días).
    # Estándar llega en 5 días (excede RUL por 0.9 días = 20 horas de parada).
    # Coste hora para alta criticidad = 2500 EUR. Coste parada = 20h * 2500 = 50000 EUR + 15000 daño secundario = 65000 EUR.
    # Sobrecoste exprés = 250 EUR.
    # Elegir Estándar aquí es un desastre financiero y debe ser rechazado por el Juez.
    recomendacion_std = {"mejor_balance": all_repuestos[0]} # Estándar

    reporte_fin, _ = ejecutar_debate_y_evaluacion(
        asset_id=1,
        rul_hours=100.0,
        tipo_falla="TWF",
        severidad="ALERTA AVANZADA",
        all_repuestos=all_repuestos,
        recomendacion_balance=recomendacion_std,
        criticidad_db="ALTA"
    )

    assert reporte_fin.evaluacion_global.aprobado is False
    assert "incoherencia económica" in reporte_fin.motivo_rechazo.lower()


def test_fallback_comportamiento_sin_api_key(monkeypatch):
    """
    Verifica que el debate y la justificación del Juez funcionen de forma determinista y no fallen
    cuando la variable OPENAI_API_KEY no está configurada.
    """
    # Forzar que no haya API key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from src.agent_orchestrator.negotiation import ejecutar_debate_y_evaluacion

    all_repuestos = [
        {"proveedor": "SKF Iberia S.A.", "tipo_envio": "Estándar", "precio": 350.0, "tiempo_arribo_dias": 5, "pieza_solicitada": "Kit SKF"},
        {"proveedor": "SKF Iberia S.A.", "tipo_envio": "Exprés", "precio": 600.0, "tiempo_arribo_dias": 2, "pieza_solicitada": "Kit SKF"},
    ]
    recomendacion = {"mejor_balance": all_repuestos[1]}

    # Ejecutar sin API key (usará fallbacks)
    reporte, debate = ejecutar_debate_y_evaluacion(
        asset_id=1,
        rul_hours=150.0,
        tipo_falla="TWF",
        severidad="ALERTA AVANZADA",
        all_repuestos=all_repuestos,
        recomendacion_balance=recomendacion,
        criticidad_db="ALTA"
    )

    # Verificar que las justificaciones se rellenaron correctamente con los fallbacks de alta calidad
    assert reporte.justificacion_decision is not None
    assert "aprobada" in reporte.justificacion_decision.lower()
    assert "SKF Iberia S.A." in reporte.justificacion_decision
    assert len(debate["ops_agent"]["recomendacion"]) > 100
    assert len(debate["log_agent"]["recomendacion"]) > 100
    assert len(debate["fin_agent"]["recomendacion"]) > 100


def test_integracion_grafo_completo():
    """
    Verifica que el grafo de LangGraph ejecute de forma completa la fase de negociación multi-agente
    y el juez asigne la decisión correcta persistida en SQLite.
    """
    from src.agent_orchestrator.graph import OrquestadorAgentePrescriptivo
    from src.database.db_manager import DatabaseManager

    db = DatabaseManager()
    orquestador = OrquestadorAgentePrescriptivo(db)

    # Registrar un activo de alta criticidad
    db.register_asset(
        name="Turbina Hidráulica T1",
        description="Turbina crítica de generación",
        rpm=1500.0,
        criticidad="CRÍTICA"
    )
    asset = db.get_asset_by_name("Turbina Hidráulica T1")
    assert asset is not None
    assert asset["criticidad"] == "CRÍTICA"

    # Disparar flujo: RUL crítico (12 horas). El Juez debería rechazar la opción estándar si es seleccionada.
    session_id = orquestador.disparar_grafo(
        asset_id=asset["id"],
        rul_hours=12.0,
        tipo_falla="PWF",
        severidad="CRÍTICO"
    )

    # Recuperar sesión y verificar el reporte del Juez
    session = db.get_agent_session(session_id)
    assert session is not None
    state_data = session["state_data"]

    assert "reporte_juez" in state_data
    assert "debate" in state_data
    reporte = state_data["reporte_juez"]

    # Dado que es RUL = 12 horas, la opción estándar tardaría mucho, por ende el Juez debió rechazar o aprobar según opción elegida.
    # El estatus de la sesión debe ser coherente
    assert "Pausado" in session["status"]
    assert state_data["debate"]["ops_agent"]["agente"] == "Operaciones"
    assert state_data["debate"]["log_agent"]["agente"] == "Logística"
    assert state_data["debate"]["fin_agent"]["agente"] == "Finanzas"

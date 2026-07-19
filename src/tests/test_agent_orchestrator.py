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
    assert status_data["status"] == "Pausado (Esperando Aprobación)"
    assert status_data["state_name"] == "aprobacion_humana"
    assert len(status_data["state_data"]["repuestos"]) == 3
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

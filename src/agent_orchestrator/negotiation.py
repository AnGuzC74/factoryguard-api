"""
Módulo de Negociación Prescriptiva Multi-Agente y Evaluación Continua (LLM-as-a-Judge).

Implementa los agentes de Operaciones, Logística y Finanzas y un Juez determinista
con soporte de generación mediante LLM real o fallback por reglas expertas estructuradas.
"""

import os
import tomllib
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import httpx

# --- Modelos Pydantic Requeridos ---

class EvaluacionGlobal(BaseModel):
    aprobado: bool = Field(description="True si la decisión es segura, económicamente óptima y técnicamente viable.")
    score_alineacion: float = Field(description="Puntaje de 0.0 a 1.0 sobre el cumplimiento de restricciones.")

class AnalisisCostos(BaseModel):
    costo_envio: float
    costo_instalacion: float
    costo_parada_estimado: float
    costo_total_solucion: float
    costo_total_escenario_falla: float

class ImpactoOperativo(BaseModel):
    criticidad_equipo: str
    horas_parada_evitadas: float
    margen_seguridad_dias: float

class ReporteJuez(BaseModel):
    evaluacion_global: EvaluacionGlobal
    analisis_costos: AnalisisCostos
    impacto_operativo: ImpactoOperativo
    justificacion_decision: str
    motivo_rechazo: Optional[str] = None
    accion_requerida: Optional[str] = None


# --- Cargador de Configuración ---

def cargar_config_negociacion(config_path: Path = Path("config.toml")) -> Dict[str, Any]:
    if not config_path.exists():
        return {
            "margen_seguridad_dias": 2.0,
            "costo_hora_critica": 5000.0,
            "costo_hora_alta": 2500.0,
            "costo_hora_media": 1000.0,
            "costo_hora_baja": 200.0,
            "danio_secundario_estimado": 15000.0
        }
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("agent_negotiation", {})


# --- Cliente OpenAI / Fallback Helper ---

def _is_cloud_mode() -> bool:
    try:
        import streamlit as st
        if st.secrets and "cloud_mode" in st.secrets:
            return bool(st.secrets["cloud_mode"])
    except Exception:
        pass
    config_neg = cargar_config_negociacion()
    return bool(config_neg.get("cloud_mode", False))

def llamar_llm_openai(prompt: str, system_prompt: str) -> Optional[str]:
    # El modo nube debe operar siempre con el generador de plantillas expertas (evitar costos de API)
    if _is_cloud_mode():
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        # Timeout razonable
        response = httpx.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=12.0)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM] Error en llamada a OpenAI: {e}")
    return None


# --- Implementación de Agentes con Enfoque Híbrido ---

def ejecutar_agente_operaciones(
    asset_id: int,
    rul_hours: float,
    criticidad: str,
    severidad: str,
    chosen_option: Dict[str, Any],
    margen_seguridad_dias: float,
    rule_1_passed: bool
) -> Dict[str, Any]:
    """
    Agente de Operaciones: Prioriza continuidad de planta, criticidad y margen de seguridad.
    """
    rul_dias = rul_hours / 24.0
    rul_limite_dias = max(0.0, rul_dias - margen_seguridad_dias)
    prov_name = chosen_option.get("proveedor", "N/A")
    tipo_envio = chosen_option.get("tipo_envio", "Estándar")
    tiempo_arribo = chosen_option.get("tiempo_arribo_dias", 0)

    # Prompt para LLM
    system_prompt = (
        "Eres el Agente de Operaciones de FactoryGuard AI, un consultor experto en mantenimiento "
        "industrial y continuidad de negocio. Tu prioridad es evitar paradas no planificadas de planta, "
        "respetar el margen de seguridad y proteger activos críticos. Hablas en español formal y profesional."
    )
    prompt = (
        f"Analiza la siguiente situación operativa y genera tu recomendación detallada:\n"
        f"- ID de Activo: {asset_id}\n"
        f"- Criticidad de Equipo: {criticidad}\n"
        f"- Severidad del Diagnóstico: {severidad}\n"
        f"- RUL Estimado: {rul_hours:.1f} horas ({rul_dias:.2f} días)\n"
        f"- Margen de Seguridad Requerido: {margen_seguridad_dias} días\n"
        f"- Opción Seleccionada: Proveedor {prov_name} con Envío {tipo_envio} (Arribo en {tiempo_arribo} días)\n"
        f"- ¿Cumple con el Margen de Seguridad?: {'Sí' if rule_1_passed else 'No'}\n\n"
        f"Genera una justificación estructurada y profesional detallando el impacto por parada no planificada "
        f"y si apruebas o exiges cambios en la logística de envío."
    )

    respuesta_llm = llamar_llm_openai(prompt, system_prompt)
    if respuesta_llm:
        recomendacion = respuesta_llm
    else:
        # Fallback determinista de alta calidad
        riesgo_texto = (
            "un nivel de riesgo operativo perfectamente acotado y seguro para la planta."
            if rule_1_passed else
            "un nivel de riesgo inaceptable. El tiempo de arribo compromete la integridad del activo antes de que se pueda intervenir de forma segura."
        )
        recomendacion = (
            f"Como Agente de Operaciones, nuestra prioridad absoluta es la continuidad operativa de la planta. "
            f"Para el activo {asset_id} (criticidad {criticidad}), estimamos un RUL restante de {rul_hours:.1f} horas ({rul_dias:.2f} días). "
            f"Considerando el margen de seguridad mandatorio de {margen_seguridad_dias} días, la ventana máxima para recibir el repuesto es de {rul_limite_dias:.2f} días.\n\n"
            f"La propuesta de {prov_name} ({tipo_envio}) con entrega en {tiempo_arribo} días representa {riesgo_texto} "
            f"{'Aprobamos la planificación actual y sugerimos preparar la orden de trabajo.' if rule_1_passed else 'Exigimos de manera urgente cambiar al modo de envío Exprés o buscar un proveedor alternativo inmediato para evitar una parada no planificada catastrófica.'}"
        )

    return {
        "agente": "Operaciones",
        "prioridad": "Continuidad de planta y margen de seguridad técnica",
        "recomendacion": recomendacion
    }


def ejecutar_agente_logistica(
    tipo_falla: str,
    chosen_option: Dict[str, Any],
    all_repuestos: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Agente de Logística: Prioriza disponibilidad de repuestos y tiempos de entrega (Lead Time).
    """
    prov_name = chosen_option.get("proveedor", "N/A")
    tipo_envio = chosen_option.get("tipo_envio", "Estándar")
    tiempo_arribo = chosen_option.get("tiempo_arribo_dias", 0)
    precio = chosen_option.get("precio", 0.0)

    # Buscar opciones del mismo proveedor para detallar en la propuesta
    opciones_prov = [r for r in all_repuestos if r.get("proveedor") == prov_name]
    opciones_texto = ""
    for opt in opciones_prov:
        opciones_texto += f"- Envío {opt.get('tipo_envio')}: {opt.get('tiempo_arribo_dias')} días, {opt.get('precio')} EUR\n"

    system_prompt = (
        "Eres el Agente de Logística de FactoryGuard AI, experto en cadena de suministro industrial "
        "y gestión de repuestos para maquinaria pesada. Tu prioridad es garantizar que el rodamiento adecuado "
        "esté disponible en el menor tiempo posible, coordinando opciones Estándar vs. Exprés. Hablas en español formal."
    )
    prompt = (
        f"Analiza las opciones logísticas y genera tu propuesta técnica de suministro:\n"
        f"- Componente requerido: Kit Rodamiento para Falla {tipo_falla}\n"
        f"- Proveedor de interés: {prov_name}\n"
        f"- Alternativas encontradas:\n{opciones_texto}\n"
        f"- Opción actual seleccionada: Envío {tipo_envio} ({tiempo_arribo} días, {precio} EUR)\n\n"
        f"Genera un informe analítico sobre el suministro, tiempos de tránsito (Lead Time) y la justificación de la ruta elegida."
    )

    respuesta_llm = llamar_llm_openai(prompt, system_prompt)
    if respuesta_llm:
        recomendacion = respuesta_llm
    else:
        # Fallback determinista
        detalles_opciones = ", ".join([f"{o.get('tipo_envio')} ({o.get('tiempo_arribo_dias')} días, {o.get('precio')} EUR)" for o in opciones_prov])
        recomendacion = (
            f"Como Agente de Logística, confirmamos la disponibilidad de repuestos para solventar la condición de {tipo_falla} en el activo. "
            f"Hemos cotizado con el proveedor de referencia {prov_name} las siguientes opciones: {detalles_opciones}.\n\n"
            f"La opción de envío {tipo_envio} garantiza el arribo en {tiempo_arribo} días con un costo logístico integrado en el precio de {precio} EUR. "
            f"Garantizamos la viabilidad de la ruta física de transporte y la asignación prioritaria de stock en los almacenes centrales."
        )

    return {
        "agente": "Logística",
        "prioridad": "Disponibilidad de repuestos, gestión de stock y Lead Time óptimo",
        "recomendacion": recomendacion
    }


def ejecutar_agente_finanzas(
    criticidad: str,
    costo_hora: float,
    chosen_option: Dict[str, Any],
    costo_parada_estimado: float,
    danio_secundario: float,
    ctc: float,
    sobrecosto_envio_expres: float,
    costo_total_falla_estandar: float,
    rule_2_passed: bool
) -> Dict[str, Any]:
    """
    Agente de Finanzas: Prioriza minimizar el Costo Total Integrado (CTC) y proteger el presupuesto.
    """
    prov_name = chosen_option.get("proveedor", "N/A")
    tipo_envio = chosen_option.get("tipo_envio", "Estándar")
    precio = chosen_option.get("precio", 0.0)

    system_prompt = (
        "Eres el Agente de Finanzas de FactoryGuard AI, un analista financiero y controlador de costos "
        "especializado en activos industriales y presupuestos de mantenimiento OPEX/CAPEX. Tu meta es proteger el "
        "presupuesto, minimizar el Costo Total Integrado (CTC) y realizar análisis costo-beneficio rigurosos. Hablas en español."
    )
    prompt = (
        f"Realiza un análisis financiero costo-beneficio para la siguiente adquisición de repuestos:\n"
        f"- Proveedor: {prov_name}\n"
        f"- Opción Elegida: Envío {tipo_envio} (Costo: {precio} EUR)\n"
        f"- Criticidad del Activo: {criticidad} (Costo de Parada por Hora: {costo_hora} EUR/h)\n"
        f"- Costo de Parada Estimado si falla: {costo_parada_estimado:.2f} EUR\n"
        f"- Daño Secundario Estimado: {danio_secundario:.2f} EUR\n"
        f"- Costo Total Integrado (CTC) del Escenario: {ctc:.2f} EUR\n"
        f"- Sobrecosto de Envío Exprés: {sobrecosto_envio_expres:.2f} EUR\n"
        f"- Pérdida Financiera Estimada con Envío Estándar si falla: {costo_total_falla_estandar:.2f} EUR\n"
        f"- ¿Cumple con la Directriz de Viabilidad Financiera?: {'Sí' if rule_2_passed else 'No'}\n\n"
        f"Genera un análisis financiero detallado defendiendo el CTC mínimo y emitiendo una postura presupuestaria firme."
    )

    respuesta_llm = llamar_llm_openai(prompt, system_prompt)
    if respuesta_llm:
        recomendacion = respuesta_llm
    else:
        # Fallback determinista
        if rule_2_passed:
            analisis = (
                f"La opción de envío {tipo_envio} es financieramente óptima. "
                f"El Costo Total Integrado de {ctc:.2f} EUR representa el mejor balance de costos para la compañía. "
                f"Dado que evitamos pérdidas operacionales severas, liberamos y aprobamos la partida presupuestaria de forma inmediata."
            )
        else:
            analisis = (
                f"¡ADVERTENCIA DE CONTROL DE RIESGOS! Rechazamos la asignación de envío Estándar. "
                f"El sobrecosto de envío Exprés es de solo {sobrecosto_envio_expres:.2f} EUR, mientras que mantener el envío estándar "
                f"nos expone a un costo de parada proyectado de {costo_parada_estimado:.2f} EUR y un daño colateral de {danio_secundario:.2f} EUR "
                f"(pérdida total potencial de {costo_total_falla_estandar:.2f} EUR). "
                f"Financieramente es imperativo realizar el pago premium del envío exprés para salvaguardar el margen neto de la operación."
            )
        recomendacion = (
            f"Como Agente de Finanzas, nuestro deber es proteger el presupuesto y minimizar el Costo Total Integrado. "
            f"Evaluamos el activo de criticidad {criticidad} con un costo de detención de {costo_hora} EUR/hora.\n\n"
            f"{analisis}"
        )

    return {
        "agente": "Finanzas",
        "prioridad": "Minimización del Costo Total Integrado (CTC) y control de OPEX",
        "recomendacion": recomendacion
    }


# --- Ejecución del Debate y Juez ---

def ejecutar_debate_y_evaluacion(
    asset_id: int,
    rul_hours: float,
    tipo_falla: str,
    severidad: str,
    all_repuestos: List[Dict[str, Any]],
    recomendacion_balance: Dict[str, Any],
    criticidad_db: Optional[str] = None
) -> ReporteJuez:
    """
    Ejecuta el debate determinista entre los tres agentes y produce la evaluación final del Juez.
    """
    # 1. Cargar parámetros desde configuración
    config_neg = cargar_config_negociacion()
    margen_seguridad_dias = config_neg.get("margen_seguridad_dias", 2.0)
    danio_secundario = config_neg.get("danio_secundario_estimado", 15000.0)

    # 2. Determinar Criticidad
    if criticidad_db and criticidad_db.strip():
        criticidad = criticidad_db.strip().upper()
    else:
        # Derivar de la severidad
        sev_upper = severidad.upper()
        if "CRÍTICO" in sev_upper or "CRITICO" in sev_upper:
            criticidad = "CRÍTICA"
        elif "ALERTA AVANZADA" in sev_upper:
            criticidad = "ALTA"
        elif "ALERTA INCIPIENTE" in sev_upper:
            criticidad = "MEDIA"
        else:
            criticidad = "BAJA"

    # 3. Obtener Costo por hora según criticidad
    costo_hora = 1000.0
    if criticidad == "CRÍTICA":
        costo_hora = config_neg.get("costo_hora_critica", 5000.0)
    elif criticidad == "ALTA":
        costo_hora = config_neg.get("costo_hora_alta", 2500.0)
    elif criticidad == "MEDIA":
        costo_hora = config_neg.get("costo_hora_media", 1000.0)
    elif criticidad == "BAJA":
        costo_hora = config_neg.get("costo_hora_baja", 200.0)

    # 4. Extraer la opción elegida por balance
    # recomendacion_balance usualmente es dict que contiene 'mejor_balance' o directamente el repuesto
    chosen_option = recomendacion_balance.get("mejor_balance") if recomendacion_balance else None
    if not chosen_option and all_repuestos:
        chosen_option = all_repuestos[0]

    if not chosen_option:
        # Fallback si no hay repuestos (ej: no requiere reemplazo)
        reporte = ReporteJuez(
            evaluacion_global=EvaluacionGlobal(aprobado=True, score_alineacion=1.0),
            analisis_costos=AnalisisCostos(
                costo_envio=0.0,
                costo_instalacion=0.0,
                costo_parada_estimado=0.0,
                costo_total_solucion=0.0,
                costo_total_escenario_falla=0.0
            ),
            impacto_operativo=ImpactoOperativo(
                criticidad_equipo=criticidad,
                horas_parada_evitadas=0.0,
                margen_seguridad_dias=margen_seguridad_dias
            ),
            justificacion_decision="No se requiere reemplazo ni adquisición de repuestos para el estado actual del activo.",
            motivo_rechazo=None,
            accion_requerida="Continuar con el monitoreo rutinario y lubricación preventiva."
        )
        debate_payload = {
            "ops_agent": {
                "agente": "Operaciones",
                "prioridad": "Continuidad de planta y margen de seguridad técnica",
                "recomendacion": "El activo está operando en parámetros normales. Se aprueba la continuidad sin intervención."
            },
            "log_agent": {
                "agente": "Logística",
                "prioridad": "Disponibilidad de repuestos, gestión de stock y Lead Time óptimo",
                "recomendacion": "No se requiere suministro de repuestos en este ciclo."
            },
            "fin_agent": {
                "agente": "Finanzas",
                "prioridad": "Minimización del Costo Total Integrado (CTC) y control de OPEX",
                "recomendacion": "OPEX óptimo sin gastos de adquisición."
            }
        }
        return reporte, debate_payload

    prov_name = chosen_option.get("proveedor", "N/A")
    tipo_envio = chosen_option.get("tipo_envio", "Estándar")
    tiempo_arribo = chosen_option.get("tiempo_arribo_dias", 0)
    precio = chosen_option.get("precio", 0.0)

    # 5. Buscar las opciones estándar vs exprés correspondientes a este proveedor para los cálculos analíticos
    prov_opts = [r for r in all_repuestos if r.get("proveedor") == prov_name]
    std_opt = next((o for o in prov_opts if o.get("tipo_envio") == "Estándar"), chosen_option)
    exp_opt = next((o for o in prov_opts if o.get("tipo_envio") == "Exprés"), chosen_option)

    # Cálculos numéricos exactos
    rul_dias = rul_hours / 24.0
    costo_instalacion = 300.0 # Costo de instalación de rodamiento estándar
    costo_envio = 250.0 if tipo_envio == "Exprés" else 0.0

    # Pérdidas si se usa la opción estándar
    tiempo_std = std_opt.get("tiempo_arribo_dias", 3)
    if tiempo_std > rul_dias:
        horas_parada_std = (tiempo_std * 24.0) - rul_hours
        costo_parada_std = horas_parada_std * costo_hora
        costo_total_falla_estandar = costo_parada_std + danio_secundario
    else:
        costo_total_falla_estandar = 0.0

    # Pérdidas para la opción elegida actualmente
    if tiempo_arribo > rul_dias:
        horas_parada_reales = (tiempo_arribo * 24.0) - rul_hours
        costo_parada_estimado = horas_parada_reales * costo_hora
        costo_total_falla_actual = costo_parada_estimado + danio_secundario
    else:
        costo_parada_estimado = 0.0
        costo_total_falla_actual = 0.0

    # Sobrecosto de Exprés
    sobrecosto_envio_expres = exp_opt.get("precio", 0.0) - std_opt.get("precio", 0.0)
    if sobrecosto_envio_expres <= 0:
        sobrecosto_envio_expres = 250.0

    # Horas de parada evitadas si elegimos una opción mejor o si evitamos el fallo por completo
    if tiempo_arribo <= rul_dias and tiempo_std > rul_dias:
        horas_parada_evitadas = (tiempo_std * 24.0) - rul_hours
    elif tiempo_arribo <= rul_dias:
        horas_parada_evitadas = 0.0 # Ninguna pérdida hubiera ocurrido
    else:
        horas_parada_evitadas = 0.0 # Sigue habiendo parada

    costo_total_solucion = precio + costo_instalacion
    costo_total_escenario_falla = costo_total_solucion + costo_total_falla_actual

    # --- Evaluación Determinista del Juez (Reglas de Negocio Estrictas) ---
    rule_1_passed = tiempo_arribo <= (rul_dias - margen_seguridad_dias)

    # Regla de viabilidad financiera: Si el costo de falla estándar supera el sobrecosto exprés, pero se eligió Estándar, se rechaza
    rule_2_passed = True
    if costo_total_falla_estandar > sobrecosto_envio_expres and tipo_envio == "Estándar":
        rule_2_passed = False

    # Aprobado global
    aprobado = rule_1_passed and rule_2_passed

    # Cálculo de score continuo de alineación
    score_alineacion = 1.0
    if not rule_1_passed:
        score_alineacion -= 0.5
    if not rule_2_passed:
        score_alineacion -= 0.5
    score_alineacion = max(0.0, score_alineacion)

    # Motivo de rechazo específico si aplica
    motivo_rechazo = None
    accion_requerida = None
    if not aprobado:
        motivos = []
        acciones = []
        if not rule_1_passed:
            motivos.append(
                f"El tiempo de arribo del repuesto ({tiempo_arribo} días) excede el margen de seguridad "
                f"técnico respecto al RUL del activo ({rul_dias:.2f} días - margen de {margen_seguridad_dias} días)."
            )
            acciones.append("Cambiar a un método de envío urgente o seleccionar un proveedor con mejor Lead Time.")
        if not rule_2_passed:
            motivos.append(
                f"Incoherencia económica severa. El costo de parada proyectado ({costo_total_falla_estandar:.2f} EUR) "
                f"supera masivamente el costo de envío exprés ({sobrecosto_envio_expres:.2f} EUR), pero se seleccionó el envío estándar."
            )
            acciones.append("Aprobar de manera mandatoria la opción de Envío Exprés para evitar paradas catastróficas.")

        motivo_rechazo = " | ".join(motivos)
        accion_requerida = " | ".join(acciones)
    else:
        accion_requerida = "Proceder con la generación de la orden de compra y programar ventana de instalación."

    # --- Generación de Justificación Conversacional con LLM / Fallback ---
    system_prompt = (
        "Eres el Agente Juez (LLM-as-a-Judge) de FactoryGuard AI, un analista senior de confiabilidad, "
        "riesgos y control financiero industrial. Tu labor es auditar las propuestas de Operaciones, Logística y Finanzas "
        "con absoluta neutralidad e imparcialidad científica y económica. Hablas en español formal de alta dirección."
    )
    prompt = (
        f"Audita el debate de negociación para la adquisición de repuestos y emite tu justificación oficial:\n"
        f"- ID de Activo: {asset_id}\n"
        f"- Criticidad de Equipo: {criticidad}\n"
        f"- RUL de Activo: {rul_hours:.1f} horas ({rul_dias:.2f} días)\n"
        f"- Margen de Seguridad: {margen_seguridad_dias} días\n"
        f"- Opción Evaluada: Proveedor {prov_name} mediante Envío {tipo_envio}\n"
        f"- Tiempo de arribo: {tiempo_arribo} días\n"
        f"- Costo de Repuesto: {precio} EUR\n"
        f"- Costo de Parada Estimado: {costo_parada_estimado:.2f} EUR\n"
        f"- Daño Secundario Proyectado: {danio_secundario:.2f} EUR\n"
        f"- Costo Total Integrado (CTC): {costo_total_escenario_falla:.2f} EUR\n"
        f"- Decisión Final del Juez: {'APROBADO' if aprobado else 'RECHAZADO'}\n"
        f"- Score de Alineación: {score_alineacion:.2f}/1.0\n"
        f"- Detalles del Incumplimiento: {motivo_rechazo if motivo_rechazo else 'Cumple todas las restricciones técnicas y económicas.'}\n\n"
        f"Genera una justificación gerencial, detallando por qué se aprueba o se rechaza, defendiendo el balance óptimo "
        f"entre continuidad operativa y eficiencia de capital."
    )

    justificacion = llamar_llm_openai(prompt, system_prompt)
    if not justificacion:
        # Fallback de plantilla de texto corporativa de alta calidad
        if aprobado:
            justificacion = (
                f"La propuesta de adquisición del repuesto con el proveedor {prov_name} bajo la modalidad de envío "
                f"{tipo_envio} ha sido APROBADA con un score de alineación de {score_alineacion:.2f}. "
                f"El análisis de confiabilidad confirma que el repuesto arribará dentro del margen seguro establecido, "
                f"evitando un impacto operativo en el activo de criticidad {criticidad}. Desde la perspectiva financiera, "
                f"el Costo Total Integrado de {costo_total_escenario_falla:.2f} EUR representa el escenario óptimo de mínimo riesgo y mínimo gasto de capital."
            )
        else:
            justificacion = (
                f"La propuesta de adquisición ha sido RECHAZADA por la auditoría debido a desviaciones críticas "
                f"respecto a las directrices de confiabilidad y control de riesgos corporativos (Score: {score_alineacion:.2f}). "
                f"Detalle técnico de la anomalía: {motivo_rechazo}. "
                f"Es imperativo reajustar los parámetros de suministro siguiendo la acción requerida: {accion_requerida}."
            )

    reporte = ReporteJuez(
        evaluacion_global=EvaluacionGlobal(aprobado=aprobado, score_alineacion=score_alineacion),
        analisis_costos=AnalisisCostos(
            costo_envio=costo_envio,
            costo_instalacion=costo_instalacion,
            costo_parada_estimado=costo_parada_estimado,
            costo_total_solucion=costo_total_solucion,
            costo_total_escenario_falla=costo_total_escenario_falla
        ),
        impacto_operativo=ImpactoOperativo(
            criticidad_equipo=criticidad,
            horas_parada_evitadas=horas_parada_evitadas,
            margen_seguridad_dias=margen_seguridad_dias
        ),
        justificacion_decision=justificacion,
        motivo_rechazo=motivo_rechazo,
        accion_requerida=accion_requerida
    )

    # 6. Ejecutar el debate en lenguaje natural para los tres agentes
    # (Los guardaremos junto con el reporte de juez para visualización del dashboard)
    ops_prop = ejecutar_agente_operaciones(
        asset_id, rul_hours, criticidad, severidad, chosen_option, margen_seguridad_dias, rule_1_passed
    )
    log_prop = ejecutar_agente_logistica(
        tipo_falla, chosen_option, all_repuestos
    )
    fin_prop = ejecutar_agente_finanzas(
        criticidad, costo_hora, chosen_option, costo_parada_estimado,
        danio_secundario, costo_total_escenario_falla, sobrecosto_envio_expres,
        costo_total_falla_estandar, rule_2_passed
    )

    # Estructura del debate para el estado
    debate_payload = {
        "ops_agent": ops_prop,
        "log_agent": log_prop,
        "fin_agent": fin_prop
    }

    return reporte, debate_payload

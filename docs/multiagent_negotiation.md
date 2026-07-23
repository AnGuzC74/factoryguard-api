# ⚖️ Sistema Multi-Agente de Negociación Prescriptiva y Evaluación Continua (LLM-as-a-Judge)

Este documento describe la especificación técnica, las ecuaciones físico-económicas y la arquitectura de orquestación implementada en **FactoryGuard AI** para transformar la generación simple de recomendaciones en un avanzado sistema de debate multi-agente y auditoría inteligente.

---

## 1. Arquitectura de Agentes Especializados

Para garantizar decisiones óptimas en la adquisición de componentes críticos (ej. rodamientos), se ha diseñado un flujo agentivo en **LangGraph** que modela un conflicto de intereses realista dentro de la planta industrial. Los nodos especializados se estructuran secuencialmente tras la recopilación de datos y la búsqueda de repuestos:

```
diagnosticar ➡️ generar_orden_prescriptiva ➡️ buscar_repuestos ➡️ presentar_comparativa
                                                                     ⬇️
finalizar_orden ⬅️ aprobacion_humana ⬅️ negociacion_multiagente (Debate + Juez)
```

Cada uno de los tres agentes defiende una métrica de rendimiento (KPI) específica:

### A. Agente Operaciones (Operations Agent)
* **Prioridad**: Continuidad de planta, mitigación de riesgos catastróficos y mantenimiento de los márgenes de seguridad técnica del activo.
* **Entradas**: Telemetría RMS de vibración, severidad de la alerta, estimación de Vida Útil Remanente ($RUL$) en horas.
* **Salidas**: Requerimiento mandatorio de ventana de mantenimiento y análisis del impacto de parada no planificada si el repuesto excede la ventana de seguridad.

### B. Agente Logística (Logistics Agent)
* **Prioridad**: Eficiencia de la cadena de suministro, disponibilidad inmediata de componentes certificados y optimización del Lead Time.
* **Entradas**: Especificación del rodamiento (ej. SKF, FAG, NSK), urgencia de entrega.
* **Salidas**: Cotización de dos alternativas de envío por proveedor (**Estándar** vs. **Exprés**), detallando costes de transporte y tiempos de tránsito precisos en días.

### C. Agente Finanzas (Finance Agent)
* **Prioridad**: Minimización del Costo Total Integrado (CTC) y protección estricta del presupuesto anual OPEX.
* **Entradas**: Coste unitario de adquisición, coste de parada no planificada por hora parametrizado por criticidad, daño secundario estimado de la máquina.
* **Salidas**: Postura presupuestaria, análisis de costo-beneficio y evaluación de la amortización del sobrecoste logístico de envío rápido contra la pérdida potencial de producción.

---

## 2. Formulaciones Físico-Económicas del Juez Determinista

El **Agente Juez (LLM-as-a-Judge)** opera bajo un enfoque robusto de **Decisión Determinista**. Esto significa que la aprobación o el rechazo de la propuesta logística se calcula mediante lógica de código pura, eliminando alucinaciones o variabilidades asociadas a modelos de lenguaje. El modelo LLM (si está disponible a través de `OPENAI_API_KEY`) se utiliza de forma exclusiva para redactar la justificación técnica en lenguaje natural.

### Ecuación 1: Conversión Temporal de RUL
Dado que las estimaciones físicas de RUL se calculan en horas, el Juez realiza la conversión lineal a días para compararla con los Lead Times logísticos:
$$RUL_{\text{días}} = \frac{RUL_{\text{horas}}}{24}$$

### Ecuación 2: Ventana Máxima de Seguridad Técnica
Para certificar una parada planificada sin riesgos mecánicos, se introduce un margen de seguridad ($M_{\text{seguridad}}$, parametrizado en `config.toml`):
$$Ventana_{\text{límite}} = RUL_{\text{días}} - M_{\text{seguridad}}$$

### Ecuación 3: Costo de Parada No Planificada
Si el tiempo de arribo del repuesto ($t_{\text{arribo}}$) es mayor que el RUL físico, la máquina fallará catastróficamente antes de la intervención, acumulando pérdidas por hora de inactividad ($C_{\text{hora\_downtime}}$):
$$C_{\text{parada}} = \max\left(0, \left(t_{\text{arribo}} \cdot 24 - RUL_{\text{horas}}\right)\right) \cdot C_{\text{hora\_downtime}}$$

### Ecuación 4: Costo Total Integrado (CTC)
El Costo Total Integrado de la solución evalúa el impacto financiero global en la planta:
$$CTC = Precio_{\text{repuesto}} + Costo_{\text{instalación}} + C_{\text{parada}} + C_{\text{daño\_secundario}}$$

Donde:
* $Precio_{\text{repuesto}}$: Coste facturado por el proveedor para el repuesto y la modalidad de envío.
* $Costo_{\text{instalación}}$: Coste estándar de mano de obra y alineación láser (fijado en 300 EUR).
* $C_{\text{daño\_secundario}}$: Pérdida por daños colaterales en el eje, estator o acoplamiento si ocurre el fallo catastrófico (fijado en 15,000 EUR según `config.toml`).

---

## 3. Criterios de Evaluación y Reglas de Rechazo del Juez

El Juez audita la propuesta seleccionada mediante dos reglas deterministas no negociables:

1. **Regla de Margen Seguro (Tiempo)**:
   La propuesta es **RECHAZADA** si el tiempo de arribo excede el límite seguro:
   $$t_{\text{arribo}} > RUL_{\text{días}} - M_{\text{seguridad}}$$

2. **Regla de Consistencia Económica (Finanzas)**:
   La propuesta es **RECHAZADA** si el costo potencial de falla de la opción estándar supera por mucho el sobrecoste logístico exprés, pero se seleccionó la opción estándar de forma injustificada:
   $$\left(C_{\text{parada\_estándar}} + C_{\text{daño\_secundario}}\right) > Sobrecoste_{\text{exprés}} \quad \land \quad Modaliad_{\text{elegida}} = \text{Estándar}$$

---

## 4. Arquitectura Híbrida de Generación (LLM + Fallback)

Para asegurar la máxima estabilidad en entornos CI/CD sin consumo de API Keys o problemas de red, el sistema implementa un robusto patrón de fallback heredado de las mejores prácticas de confiabilidad:

* **Modo Online**: Si existe la variable de entorno `OPENAI_API_KEY`, el orquestador realiza llamadas HTTP optimizadas a través de `httpx` hacia `gpt-4o-mini` con `temperature: 0.0`. Los prompts inyectan el contexto del debate y los datos numéricos exactos de costes, y el LLM responde generando argumentos y posturas altamente realistas y profesionales para cada agente y para el dictamen del Juez.
* **Modo Offline**: Si la clave API no está disponible, el sistema utiliza generadores basados en plantillas de texto ricas que autocomplementan las justificaciones técnicas y financieras de forma coherente según los datos calculados. **No hay fallos silenciosos ni caídas del sistema.**

---

## 5. Integración Visual en el Dashboard de Streamlit

La interfaz gráfica del dashboard (`src/app/dashboard.py`) incorpora una pestaña y sección dedicada **"🤖 Orquestación Prescriptiva Multi-Agente"**:

1. **El Debate Visual**: Muestra las propuestas de Operaciones (Azul), Logística (Morado) y Finanzas (Rosa) en tres columnas independientes con bordes distintivos y prioridades explícitas.
2. **Alertas Color-Coded**: Las aprobaciones del Juez se indican mediante banners verdes de éxito, y los rechazos mediante banners rojos de advertencia que detallan explícitamente el **motivo de rechazo** y la **acción requerida**.
3. **Tabla de Costos**: Desglosa numéricamente el impacto financiero de la compra (instalación, envío, parada y el CTC consolidado) en un formato tabular limpio y estructurado.
4. **Control Humano Obligatorio**: Incluye botones interactivos de aprobación y rechazo manual para que el operador de planta ejerza la última palabra (Human-in-the-loop). El estatus de la sesión se guarda en tiempo real en la base de datos de SQLite (`agent_sessions`).

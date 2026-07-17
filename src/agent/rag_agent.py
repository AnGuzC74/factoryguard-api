"""
Agente de IA con RAG (Retrieval-Augmented Generation) para el sistema industrial.
Utiliza ChromaDB para búsqueda semántica de manuales y genera recomendaciones
basadas en reglas expertas (sin LLM para evitar dependencias externas).
"""
from pathlib import Path
from typing import List, Dict, Any
import tomllib
import sys

try:
    import chromadb
    from chromadb.utils import embedding_functions
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class RAGAgent:
    def __init__(self, config_path: Path = Path("config.toml")):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

        self.rag_config = self.config.get("rag", {})
        self.chroma_path = self.rag_config.get("chroma_db_path", "datos/chroma_db")
        self.collection_name = self.rag_config.get("collection_name", "manuales_mantenimiento")
        self.embeddings_model = self.rag_config.get("embeddings_model", "sentence-transformers/all-MiniLM-L6-v2")

        self.umbral_critico = self.config["umbrales_severidad"]["critico_rms"]
        self.umbral_alerta = self.config["umbrales_severidad"]["alerta_rms"]

        self.client = None
        self.collection = None
        self._initialize_chroma()

    def _initialize_chroma(self) -> None:
        if not RAG_AVAILABLE:
            return
        try:
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embeddings_model
                )
            )
            print(f"[RAG] ChromaDB inicializado en {self.chroma_path}")
            
            # Autoseeding si la colección está vacía para soportar despliegues instantáneos en la nube
            if self.collection.count() == 0:
                print("[RAG] Sembrando base de datos vectorial con manuales técnicos...")
                documentos = [
                    "Procedimiento de reemplazo de rodamiento con falla en Pista Externa (BPFO). Para cambiar un rodamiento con daños en la pista externa (BPFO), detenga el motor por completo, aplique LOTO (Bloqueo y Etiquetado), use un extractor mecánico o hidráulico de garras para retirar el rodamiento viejo, limpie la zona del eje con solvente dieléctrico, caliente el rodamiento nuevo por inducción a 110°C y móntelo en caliente aplicando presión únicamente en el anillo que tiene el ajuste. (Fuente: Manual SKF de Montaje - Sección 4.2)",
                    "Inspección y monitoreo de rodamiento con desgaste en Pista Interna (BPFI). Las fallas en la pista interna (BPFI) generan impactos periódicos de alta energía. Se recomienda programar un análisis de ultrasonido acústico de alta frecuencia y aumentar provisionalmente la lubricación con grasa para altas temperaturas para amortiguar el impacto metálico hasta que se pueda programar la parada técnica. (Fuente: Procedimiento de Mantenimiento Preventivo Alfonzo Rivas - Sección 3.1)",
                    "Procedimiento de reemplazo de rodamiento con falla en Elementos Rodantes (BSF). El daño en las bolas o elementos rodantes (BSF) provoca inestabilidad rotacional severa. Retire la tapa del soporte del rodamiento, verifique si hay picaduras o decoloración azulada por sobrecalentamiento en los rodillos y reemplace la unidad completa de inmediato si se detecta pérdida de lubricación crítica. (Fuente: Manual de Ingeniería FAG - Sección 6.8)",
                    "Mantenimiento preventivo estándar para rodamientos y alineación de ejes. Un rodamiento saludable requiere una alineación láser de acoplamientos con tolerancia menor a 0.05 mm y una lubricación balanceada. Registre periódicamente las lecturas térmicas para correlacionar con los valores RMS de vibración. (Fuente: Guía Práctica de Confiabilidad Industrial - Sección 1.1)",
                    "Procedimiento para fallas de jaula (FTF) e inestabilidad de la jaula. La falla de jaula (FTF) es una de las más peligrosas ya que puede causar la rotura instantánea y el bloqueo del eje. Ante alarmas en la frecuencia FTF, programe un paro técnico de emergencia de inmediato. (Fuente: Manual de Seguridad Operativa - Sección 2.4)"
                ]
                metadatas = [
                    {"fuente": "Manual de Montaje SKF", "seccion": "Sección 4.2", "tema": "BPFO"},
                    {"fuente": "Manual Alfonzo Rivas", "seccion": "Sección 3.1", "tema": "BPFI"},
                    {"fuente": "Manual de Ingeniería FAG", "seccion": "Sección 6.8", "tema": "BSF"},
                    {"fuente": "Guía de Confiabilidad", "seccion": "Sección 1.1", "tema": "Sano"},
                    {"fuente": "Manual de Seguridad Operativa", "seccion": "Sección 2.4", "tema": "FTF"}
                ]
                ids = [f"doc_manual_{i}" for i in range(len(documentos))]
                self.collection.add(
                    documents=documentos,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"[RAG] Sembrado completado con éxito. {self.collection.count()} documentos cargados.")
        except Exception as e:
            print(f"[RAG] Error inicializando/sembrando ChromaDB: {e}")
            self.client = None
            self.collection = None

    def query(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        if not self.collection:
            return []
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            docs = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    docs.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else None
                    })
            return docs
        except Exception as e:
            print(f"[RAG] Error en consulta: {e}")
            return []

    def generar_recomendacion(self, asset_name: str, status: Dict) -> str:
        zona = status.get('zona_falla', '')
        rul = status.get('rul_hours', 999)

        if rul < 10:
            query = f"Procedimiento de reemplazo de rodamiento con falla en {zona}"
        elif rul < 50:
            query = f"Inspección y monitoreo de rodamiento con desgaste en {zona}"
        else:
            query = "Mantenimiento preventivo estándar para rodamientos"

        docs = self.query(query, n_results=2)

        recomendacion = self._generar_recomendacion_base(status)

        if docs and RAG_AVAILABLE:
            recomendacion += "\n\n📚 **Información de referencia técnica:**"
            for i, doc in enumerate(docs, 1):
                content = doc['content'][:250] + "..." if len(doc['content']) > 250 else doc['content']
                recomendacion += f"\n\n{i}. {content}"
                if doc['metadata']:
                    fuente = doc['metadata'].get('fuente', 'Manual')
                    seccion = doc['metadata'].get('seccion', '')
                    recomendacion += f"\n   *(Fuente: {fuente}{f' - {seccion}' if seccion else ''})*"

        return recomendacion

    def _generar_recomendacion_base(self, status: Dict) -> str:
        rms_max = status.get('rms_max', 0)
        rul = status.get('rul_hours', 999)
        zona = status.get('zona_falla', '')

        if rms_max >= self.umbral_critico:
            return f"""🚨 **URGENTE - REEMPLAZO INMEDIATO**

El rodamiento ha alcanzado el umbral crítico. Se requiere acción inmediata.

**Falla detectada:** {zona}
**Acción:** Reemplazar rodamiento antes de la próxima operación.
**Riesgo:** Parada no planificada en las próximas horas."""

        elif rms_max >= self.umbral_alerta:
            if rul < 50:
                return f"""⚠️ **ALERTA AVANZADA - PROGRAMAR REEMPLAZO**

El rodamiento muestra desgaste significativo. Planificar reemplazo en el corto plazo.

**Falla detectada:** {zona}
**RUL estimado:** {rul:.1f} horas
**Acción:** Programar reemplazo en la próxima ventana de mantenimiento."""
            else:
                return f"""🟡 **ALERTA INCIPIENTE - MONITOREAR**

Se detecta inicio de desgaste. Continuar con monitoreo periódico.

**Falla detectada:** {zona}
**RUL estimado:** {rul:.1f} horas
**Acción:** Registrar tendencia y planificar inspección visual."""

        else:
            return f"""🟢 **OPERACIÓN NORMAL**

El rodamiento se encuentra dentro de los parámetros operativos.

**Estado:** Saludable
**Acción:** Continuar con monitoreo de rutina."""

    def responder_conversacional(self, query_text: str) -> str:
        """
        Genera una respuesta conversacional y experta orientada a un chatbot
        combinando heurísticas de lenguaje, definiciones físicas y consultas RAG en ChromaDB.
        """
        # 1. Normalizar query
        q = query_text.lower().strip()
        
        # 2. Manejo de saludos y cortesías
        saludos = ["hola", "buenos dias", "buenas tardes", "buenas noches", "saludos", "que tal", "como estas", "buenos días"]
        if any(s in q for s in saludos):
            return """👋 ¡Hola! Soy tu asistente de Inteligencia Artificial especializado en confiabilidad y diagnóstico de rodamientos de FactoryGuard AI. 

Puedo ayudarte a resolver dudas sobre:
- Interpretación de métricas de vibraciones (**RMS**, **FFT**, **Punto de Inflexión**).
- Frecuencias de falla teóricas de rodamientos (**BPFO**, **BPFI**, **BSF**, **FTF**).
- Procedimientos de mantenimiento, montaje o lubricación (según manuales SKF, FAG, etc.).

¿En qué puedo colaborar contigo hoy?"""
            
        cortesias = ["gracias", "muchas gracias", "excelente", "perfecto", "entendido", "ok", "buenisimo", "genial", "gracias!"]
        if any(c == q or q.startswith(c) for c in cortesias):
            return """¡De nada! Es un placer ayudarte. Recuerda que la monitorización continua y la analítica predictiva son las claves para evitar paradas no planificadas. Si tienes más dudas sobre rodamientos o análisis espectral, aquí estaré para asistirte."""

        # 3. Consultas sobre conceptos físicos (Frecuencias, RMS, RUL)
        if "bpfo" in q:
            return """🌀 **BPFO (Ball Pass Frequency Outer Race)** representa la frecuencia característica de paso de elementos rodantes sobre un defecto localizado en la **pista externa** del rodamiento.

- **Fórmula:** $BPFO = \\frac{N_b}{2} \\cdot f_r \\cdot \\left(1 - \\frac{B_d}{P_d} \\cos\\phi\\right)$
- **Comportamiento:** Genera impactos periódicos repetitivos de alta energía. Es la falla más común y fácil de detectar mediante análisis espectral y envolvente de vibraciones.
- **Acción sugerida:** Según las guías de montaje, al superarse el umbral de alerta se debe planificar el reemplazo de la unidad antes de cruzar la zona crítica para evitar daños severos en el eje."""

        if "bpfi" in q:
            return """🌀 **BPFI (Ball Pass Frequency Inner Race)** representa la frecuencia de paso de elementos rodantes sobre un defecto localizado en la **pista interna** del rodamiento.

- **Fórmula:** $BPFI = \\frac{N_b}{2} \\cdot f_r \\cdot \\left(1 + \\frac{B_d}{P_d} \\cos\\phi\\right)$
- **Comportamiento:** Como el defecto está en la pista interna (que gira solidaria al eje), la zona dañada entra y sale de la zona de carga de la máquina, lo que modula la amplitud de la señal de vibración con las bandas laterales a la frecuencia de rotación del eje ($f_r$).
- **Acción sugerida:** Monitorear de cerca mediante demodulación de envolvente, ya que la propagación del daño suele ser más veloz que en la pista externa debido a los esfuerzos de contacto del eje."""

        if "bsf" in q:
            return """🌀 **BSF (Ball Spin Frequency)** es la frecuencia de giro del **elemento rodante** (bolas o rodillos) sobre su propio eje.

- **Fórmula:** $BSF = \\frac{P_d}{2 \\cdot B_d} \\cdot f_r \\cdot \\left(1 - \\left(\\frac{B_d}{P_d} \\cos\\phi\\right)^2\\right)$
- **Comportamiento:** Los defectos en las bolas generan impactos dobles cuando hacen contacto tanto con la pista interna como con la externa en cada revolución de la bola. En el espectro se observan típicamente la BSF y sus harmónicos modulados por la velocidad de la jaula (FTF).
- **Acción sugerida:** Incrementar la frecuencia de inspección y lubricación con grasas adecuadas de alta viscosidad para mitigar el desgaste por deslizamiento."""

        if "ftf" in q:
            return """🌀 **FTF (Fundamental Train Frequency)** representa la frecuencia de rotación de la **jaula porta-bolas**.

- **Fórmula:** $FTF = \\frac{f_r}{2} \\cdot \\left(1 - \\frac{B_d}{P_d} \\cos\\phi\\right)$
- **Comportamiento:** Frecuencia típicamente muy baja (menor a la frecuencia de rotación del eje $f_r$, usualmente en torno a los $10 - 15\\text{ Hz}$). Las alarmas en la FTF indican inestabilidad de la jaula o daño estructural en la misma.
- **Acción sugerida:** **Crítica**. La rotura de la jaula causa el bloqueo instantáneo del rodamiento, pudiendo provocar la rotura catastrófica del eje o daños graves en el estator del motor. Se recomienda programar parada de emergencia."""

        if "rms" in q:
            return """📊 El **valor RMS (Root Mean Square)** o valor eficaz de la vibración es el indicador global primario de severidad mecánica (bajo normas como la **ISO 10816**).

- **Ecuación:** $RMS = \\sqrt{\\frac{1}{N}\\sum_{n=1}^{N} \\bar{x}[n]^2}$
- **Significado:** Representa la energía vibratoria total contenida en la señal temporal de vibración.
- **Limitación:** Al ser un promedio energético, las fallas incipientes de baja energía (pequeñas picaduras o microfisuras iniciales) a menudo no alteran significativamente el RMS global, por lo que es necesario combinarlo con análisis de frecuencia (FFT) y demodulación de envolvente para un diagnóstico predictivo completo."""

        if "rul" in q or "vida util" in q or "vida útil" in q:
            return """📈 **RUL (Remaining Useful Life)** o Vida Útil Remanente es la estimación del tiempo o ciclos que le quedan a un componente (en este caso, un rodamiento) antes de cruzar su umbral crítico de falla irreversible ($0.25\\text{ g}$).

- **En FactoryGuard AI**: Empleamos un **motor híbrido adaptativo** de pronóstico. Evalúa de forma dinámica modelos de regresión lineal y regresión log-exponencial sobre el historial de RMS actualizados en tiempo real.
- **Válvula de Seguridad**: El sistema cuenta con lógica de irreversibilidad para evitar que fluctuaciones transitorias del RMS falseen un incremento ficticio del RUL.
- **Filtro de Apagado**: Si la máquina se detiene, el cálculo de RUL se suspende automáticamente para no distorsionar el historial analítico predictivo."""

        # 4. Búsqueda semántica usando ChromaDB para otras consultas técnicas
        docs = self.query(query_text, n_results=2)
        if docs and RAG_AVAILABLE:
            respuesta = f"🔍 **Análisis Técnico de Referencia (Base de Conocimiento RAG)**:\n\nBasándome en los manuales de ingeniería y guías de mantenimiento de FactoryGuard AI, he encontrado la siguiente información de alta relevancia para tu consulta:\n"
            for i, doc in enumerate(docs, 1):
                content = doc['content']
                fuente = doc['metadata'].get('fuente', 'Manual Técnico')
                seccion = doc['metadata'].get('seccion', '')
                respuesta += f"\n📖 **Ref {i}** ({fuente} - {seccion}):\n> {content}\n"
            respuesta += "\n*¿Deseas profundizar en algún paso específico de este procedimiento o en el cálculo de las frecuencias de falla asociadas?*"
            return respuesta

        # 5. Respuesta fallback profesional
        return f"""🤔 Recibí tu consulta sobre: *\"{query_text}\"*. 

Como experto en mantenimiento predictivo e IA de FactoryGuard AI, te comento que para darte la respuesta más detallada y estructurada posible, puedes formularme preguntas relacionadas con:
- **Conceptos**: RMS, RUL, FFT, Microcrack, Inflexión Log.
- **Frecuencias cinemáticas de falla**: BPFO, BPFI, BSF, FTF.
- **Procedimientos de mantenimiento**: Reemplazo de pista externa, monitoreo de pista interna, alineación láser o lubricación de elementos rodantes.

¿Cuál de estos temas te gustaría que analicemos en detalle hoy?"""

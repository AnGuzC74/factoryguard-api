"""
Agente de IA con RAG (Retrieval-Augmented Generation) para el sistema industrial.
Utiliza ChromaDB para búsqueda semántica de manuales y genera recomendaciones
basadas en reglas expertas (sin LLM para evitar dependencias externas).
"""
from pathlib import Path
from typing import List, Dict, Any
import tomllib

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
        except Exception as e:
            print(f"[RAG] Error inicializando ChromaDB: {e}")
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
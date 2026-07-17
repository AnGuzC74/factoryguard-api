import streamlit as st
import sys
from pathlib import Path

# Agregar directorios para compatibilidad absoluta en Streamlit Cloud
root_dir = str(Path(__file__).parent.parent.parent.parent)
src_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Forzar la recarga del módulo en Streamlit Cloud para evitar que use la memoria caché vieja de sys.modules
for key in list(sys.modules.keys()):
    if "rag_agent" in key or "agent" in key:
        sys.modules.pop(key, None)

try:
    from src.agent.rag_agent import RAGAgent
except ImportError:
    from agent.rag_agent import RAGAgent

def main():
    st.title("💬 Chat con el Agente RAG")
    st.caption("Pregunta sobre rodamientos, análisis de vibraciones, BPFO, BPFI, etc.")

    # Inicializar agente
    if "rag_agent" not in st.session_state:
        try:
            st.session_state.rag_agent = RAGAgent()
        except Exception as e:
            st.session_state.rag_agent = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hola, soy tu asistente experto en rodamientos. ¿En qué puedo ayudarte?"}
        ]

    # Mostrar historial
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Entrada del usuario
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Respuesta del agente
        with st.spinner("Consultando base de conocimiento..."):
            try:
                if st.session_state.rag_agent is not None:
                    response = st.session_state.rag_agent.responder_conversacional(prompt)
                else:
                    response = "Lo siento, el agente RAG no pudo inicializarse porque la base de datos ChromaDB no está lista."
            except Exception as e:
                response = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    try:
        st.set_page_config(page_title="Chat RAG", page_icon="💬", layout="centered")
    except Exception:
        pass  # Evitar error si set_page_config ya fue llamado por el script principal
    main()

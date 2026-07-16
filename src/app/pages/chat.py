import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.agent.rag_agent import RAGAgent

st.set_page_config(page_title="Chat RAG", page_icon="💬", layout="centered")

st.title("💬 Chat con el Agente RAG")
st.caption("Pregunta sobre rodamientos, análisis de vibraciones, BPFO, BPFI, etc.")

# Inicializar agente
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = RAGAgent()

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
            status = {
                "rms_actual": 0.0,
                "rms_max": 0.0,
                "frecuencia": 0.0,
                "rul_hours": 999,
                "zona_falla": "No definida"
            }
            response = st.session_state.rag_agent.generar_recomendacion("Usuario", status)
        except Exception as e:
            response = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
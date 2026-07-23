"""
Pruebas de Verificación de Memoria y Carga Dinámica de Librerías en Cloud Mode.
"""

import subprocess
import sys
import pytest


def test_cloud_mode_imports_not_loaded():
    """
    Verifica mediante un subproceso limpio que chromadb y sentence_transformers
    NO se cargan en sys.modules cuando cloud_mode=True.
    """
    python_code = """
import sys
import streamlit as st
# Inyectar secreto de cloud_mode
st.secrets = {"cloud_mode": True}

from src.agent.rag_agent import RAGAgent
agent = RAGAgent()

chromadb_loaded = "chromadb" in sys.modules
st_loaded = "sentence_transformers" in sys.modules

print(f"chromadb_loaded:{chromadb_loaded}")
print(f"sentence_transformers_loaded:{st_loaded}")
"""
    result = subprocess.run(
        [sys.executable, "-c", python_code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "."}
    )

    assert result.returncode == 0
    assert "chromadb_loaded:False" in result.stdout
    assert "sentence_transformers_loaded:False" in result.stdout


def test_local_mode_imports_are_loaded():
    """
    Verifica mediante un subproceso limpio que chromadb y sentence_transformers
    SÍ se cargan en sys.modules cuando cloud_mode=False.
    """
    python_code = """
import sys
import streamlit as st
# Inyectar secreto de cloud_mode = False
st.secrets = {"cloud_mode": False}

from src.agent.rag_agent import RAGAgent
agent = RAGAgent()

chromadb_loaded = "chromadb" in sys.modules
st_loaded = "sentence_transformers" in sys.modules

print(f"chromadb_loaded:{chromadb_loaded}")
print(f"sentence_transformers_loaded:{st_loaded}")
"""
    result = subprocess.run(
        [sys.executable, "-c", python_code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "."}
    )

    assert result.returncode == 0
    assert "chromadb_loaded:True" in result.stdout
    assert "sentence_transformers_loaded:True" in result.stdout

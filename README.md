# 🏭 Industrial AI Prognostics System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![RAG](https://img.shields.io/badge/RAG-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![QA](https://img.shields.io/badge/QA-Passing-brightgreen.svg)](scripts/run_qa.py)

**Sistema completo de pronóstico de vida útil (RUL) para rodamientos industriales con arquitectura de microservicios, RAG, alertas automáticas y base de datos multi-activo.**

---

## 📊 Diagrama de Arquitectura

```mermaid
flowchart TD
    A[Usuario] --> B{Orquestador (run.py)} B -->
    B --> C[Pipeline de Ingesta<br>Multiprocesamiento]
    B --> D[Dashboard Streamlit<br>Sin semáforo]
    B --> E[API REST FastAPI]
    B --> F[Reportes PDF/Rich]

    C --> G[Archivos .txt NASA]
    G --> H[Parquet + CSV]
    H --> I[SQLite Multi-Activo]

    I --> J[Motor RUL Híbrido<br>Lineal/Exponencial]
    I --> K[Alertas Email/Slack]
    I --> L[Agente RAG ChromaDB]

    L --> M[Manuales Técnicos]
    M --> N[Recomendaciones Expertas]

    E --> I
    F --> I
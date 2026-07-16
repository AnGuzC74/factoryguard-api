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
    A[Usuario] --> B[Orquestador run.py]
    B --> C[Pipeline de Ingesta]
    B --> D[Dashboard Streamlit]
    B --> E[API REST FastAPI]
    B --> F[Reportes PDF]
    C --> G[Archivos NASA]
    G --> H[Parquet + CSV]
    H --> I[SQLite Multi-Activo]
    I --> J[Motor RUL Hibrido]
    I --> K[Alertas]
    I --> L[Agente RAG ChromaDB]
    L --> M[Manuales Tecnicos]
    M --> N[Recomendaciones Expertas]
    E --> I
    F --> I

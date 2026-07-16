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

---

## 1. Descripción del Sistema

FactoryGuard API es un microservicio industrial para el diagnóstico de fallos en rodamientos de alta velocidad mediante análisis de vibraciones. El sistema procesa señales crudas de acelerómetros, aplica técnicas de procesamiento digital de señales (DSP), y emite un diagnóstico estructurado que incluye estado de salud, tipo de fallo, severidad y tiempo de vida útil remanente (RUL).

El sistema opera sobre el dataset NASA IMS Bearing, que documenta la degradación acelerada de rodamientos bajo carga controlada. Los datos se ingieren desde archivos de texto crudos, se transforman a formatos columnares (Parquet, CSV) y se almacenan en SQLite para consulta multi-activo.

## 2. Arquitectura de Procesamiento

El flujo de datos sigue una arquitectura secuencial con cuatro etapas:

1. **Ingesta**: lectura de archivos .txt del dataset NASA IMS mediante pipeline con multiprocesamiento
2. **Transformación**: cómputo de métricas de dominio (RMS, FFT, frecuencia dominante) y almacenamiento en Parquet y CSV
3. **Análisis**: motor de RUL híbrido (lineal/exponencial), aislamiento cinemático de defectos, válvula check de seguridad
4. **Exposición**: API REST (FastAPI), dashboard interactivo (Streamlit) y agente RAG (ChromaDB) para consulta de manuales técnicos

## 3. Dependencias Técnicas

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.14 |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| DSP | NumPy, SciPy |
| Datos | Polars, SQLite, Parquet |
| Agente | ChromaDB + LangGraph |
| Monitoreo | Evidently |
| Reportes | ReportLab, Rich |
| Infraestructura | Docker, GitHub Actions |

## 4. Instalación

```bash
git clone https://github.com/AnGuzC74/factoryguard-api.git
cd factoryguard-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py

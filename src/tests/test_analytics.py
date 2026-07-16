import numpy as np
import polars as pl
import pytest
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.core.dsp import (
    remover_dc_offset,
    calcular_rms,
    ejecutar_fft,
    calcular_rul_hibrido,
    calcular_punto_inflexion_log
)
from src.app.dashboard import DashboardPrognosisIndustrial


def test_remocion_dc_offset():
    senal = np.array([6.0, 4.0, 6.0, 4.0, 5.0])
    senal_centrada = remover_dc_offset(senal)
    assert np.isclose(np.mean(senal_centrada), 0.0, atol=1e-7)


def test_calculo_rms_onda_pura():
    fs = 20000
    t = np.arange(16384) / fs
    senal = 2.0 * np.sin(2 * np.pi * 100 * t)
    rms_calculado = calcular_rms(senal)
    rms_teorico = 2.0 / np.sqrt(2)
    assert np.isclose(rms_calculado, rms_teorico, rtol=1e-3)


def test_precision_espectral_fft():
    fs = 20000
    n = 16384
    t = np.arange(n) / fs
    freq_inyectada = 150.0
    senal = 1.5 * np.sin(2 * np.pi * freq_inyectada * t)
    freqs, amps = ejecutar_fft(senal, fs)
    idx_max = np.argmax(amps)
    assert np.abs(freqs[idx_max] - freq_inyectada) <= 1.22


def test_rul_hibrido_lineal():
    x = np.arange(0, 100)
    y = 0.05 + 0.002 * x
    ciclos, _, _, modelo = calcular_rul_hibrido(x, y, umbral_critico=0.25)
    assert modelo == "Lineal"
    assert ciclos < 100


def test_rul_hibrido_exponencial():
    x = np.arange(0, 100)
    y = 0.05 * np.exp(0.01 * np.arange(100))
    ciclos, _, _, modelo = calcular_rul_hibrido(x, y, umbral_critico=0.25)
    assert modelo == "Exponencial"
    assert ciclos < 200


def test_punto_inflexion():
    x = np.arange(0, 100)
    y_rms = np.zeros(100)
    y_rms[0:70] = 0.05
    y_rms[70:100] = 0.05 * np.exp(0.05 * np.arange(0, 30))
    punto = calcular_punto_inflexion_log(x, y_rms, ventana=5)
    assert punto is not None
    assert 65 < punto < 75


def test_verificar_armonicos():
    dash = DashboardPrognosisIndustrial()
    dash.TOLERANCIA_ARMONICA = 2.0
    assert dash.verificar_armonicos(473.0, 236.4, max_orden=3) is True
    assert dash.verificar_armonicos(480.0, 236.4, max_orden=3) is False


@pytest.mark.stress
def test_estres_api_local():
    try:
        import requests
    except ImportError:
        pytest.skip("requests no instalado")

    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        health.raise_for_status()
    except (requests.ConnectionError, requests.Timeout):
        pytest.skip("API no disponible. Ejecuta 'docker-compose up' antes de esta prueba.")

    archivos = [
        "2004.02.12.10.32.39",
        "2004.02.18.20.42.39",
        "2004.02.19.06.22.39"
    ]

    tiempos = []
    for archivo in archivos * 10:
        t0 = time.time()
        try:
            r = requests.post(
                "http://localhost:8000/predict",
                params={"archivo_origen": archivo},
                timeout=5
            )
            r.raise_for_status()
            tiempos.append(time.time() - t0)
        except Exception as e:
            print(f"Error en {archivo}: {e}")

    assert len(tiempos) > 0, "No se completó ninguna petición"
    latencia_promedio_ms = np.mean(tiempos) * 1000
    assert latencia_promedio_ms < 500, f"Latencia media: {latencia_promedio_ms:.0f}ms (límite: 500ms)"
    print(f"✅ Stress test: {len(tiempos)} consultas, media {latencia_promedio_ms:.0f}ms")
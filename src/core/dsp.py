import numpy as np
from scipy.fft import fft, fftfreq
from typing import Tuple, Union


def remover_dc_offset(senal: np.ndarray) -> np.ndarray:
    return senal - np.mean(senal)


def calcular_rms(senal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(senal**2)))


def ejecutar_fft(senal: np.ndarray, frecuencia_muestreo: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(senal)
    if n == 0:
        return np.array([]), np.array([])
    senal_centrada = remover_dc_offset(senal)
    componentes = fft(senal_centrada)
    frecuencias_totales = fftfreq(n, 1 / frecuencia_muestreo)
    mitad = n // 2
    frecuencias = frecuencias_totales[:mitad]
    amplitudes = (2.0 / n) * np.abs(componentes[:mitad])
    return frecuencias, amplitudes


def encontrar_pico_espectral(senal: np.ndarray, frecuencia_muestreo: int) -> Tuple[float, float]:
    freqs, amps = ejecutar_fft(senal, frecuencia_muestreo)
    if len(amps) == 0:
        return 0.0, 0.0
    idx_max = np.argmax(amps)
    return freqs[idx_max], amps[idx_max]


def calcular_punto_inflexion_log(x: np.ndarray, y_rms: np.ndarray, ventana: int = 5) -> Union[int, None]:
    y_log = np.log(np.clip(y_rms, 1e-4, None))
    if len(y_log) < ventana * 2:
        return None
    pendientes = np.diff(y_log) / np.diff(x)
    media_pend = np.mean(pendientes)
    std_pend = np.std(pendientes)
    if std_pend < 1e-9:
        return None
    puntos_criticos = np.where(pendientes > media_pend + 1.5 * std_pend)[0]
    if len(puntos_criticos) > 0:
        return int(x[puntos_criticos[0] + 1])
    return None


def calcular_rul_hibrido(
    x_ciclos: np.ndarray,
    y_rms: np.ndarray,
    umbral_critico: float,
    horizonte_prediccion: int = 300,
    min_datos: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, str]:
    if len(x_ciclos) < min_datos:
        return 999.0, np.array([]), np.array([]), "Datos insuficientes"

    n_ventana = min(len(x_ciclos), 100)
    x = x_ciclos[-n_ventana:]
    y = y_rms[-n_ventana:]

    x_centrado = x - x[-1]

    # Modelo Lineal
    coeff_lin = np.polyfit(x_centrado, y, 1)
    m_lin, b_lin = coeff_lin
    y_pred_lin = m_lin * x_centrado + b_lin
    rmse_lin = np.sqrt(np.mean((y - y_pred_lin) ** 2))

    # Modelo Exponencial
    y_clipped = np.clip(y, 1e-4, None)
    y_log = np.log(y_clipped)
    coeff_exp = np.polyfit(x_centrado, y_log, 1)
    m_exp, b_exp = coeff_exp
    y_pred_exp = np.exp(b_exp) * np.exp(m_exp * x_centrado)
    rmse_exp = np.sqrt(np.mean((y - y_pred_exp) ** 2))

    x_futuro_rel = np.arange(1, horizonte_prediccion + 1)
    x_futuro_abs = x[-1] + x_futuro_rel

    # --- CRITERIO DE SELECCIÓN MEJORADO ---
    # Si la pendiente exponencial es positiva y la lineal no, forzar exponencial
    if m_exp > 0 and m_lin <= 0:
        modelo = "Exponencial"
        y_proyectado = np.exp(b_exp) * np.exp(m_exp * x_futuro_rel)
        if m_exp <= 0:
            return 999.0, x_futuro_abs, y_proyectado, modelo
        indices_falla = np.where(y_proyectado >= umbral_critico)[0]
        if len(indices_falla) == 0:
            return 999.0, x_futuro_abs, y_proyectado, modelo
        return float(indices_falla[0]), x_futuro_abs, y_proyectado, modelo

    # Caso normal: elegir por RMSE
    if rmse_lin <= rmse_exp:
        modelo = "Lineal"
        y_proyectado = b_lin + m_lin * x_futuro_rel
        y_proyectado = np.clip(y_proyectado, 0, None)
        if m_lin <= 0:
            return 999.0, x_futuro_abs, y_proyectado, modelo
        indices_falla = np.where(y_proyectado >= umbral_critico)[0]
        if len(indices_falla) == 0:
            return 999.0, x_futuro_abs, y_proyectado, modelo
        return float(indices_falla[0]), x_futuro_abs, y_proyectado, modelo
    else:
        modelo = "Exponencial"
        y_proyectado = np.exp(b_exp) * np.exp(m_exp * x_futuro_rel)
        if m_exp <= 0:
            return 999.0, x_futuro_abs, y_proyectado, modelo
        indices_falla = np.where(y_proyectado >= umbral_critico)[0]
        if len(indices_falla) == 0:
            return 999.0, x_futuro_abs, y_proyectado, modelo
        return float(indices_falla[0]), x_futuro_abs, y_proyectado, modelo
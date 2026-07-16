import time
import tomllib
import logging
from pathlib import Path
import numpy as np
import polars as pl
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.dsp import ejecutar_fft, calcular_rms

logger = logging.getLogger("motor_analisis")

class MotorAnalisisVibraciones:
    def __init__(self, ruta_config: Path = Path("config.toml")):
        with open(ruta_config, "rb") as f:
            self.config = tomllib.load(f)
        self.archivo_parquet = Path(self.config["infraestructura"]["archivo_parquet"])
        self.frecuencia_muestreo = self.config["fisica_rodamiento"]["frecuencia_muestreo"]
        self.puntos_fft = self.config["fisica_rodamiento"]["puntos_fft"]
        self.umbral_cv_rpm = self.config["fisica_rodamiento"]["umbral_estabilidad_rpm"]
        self.archivo_telemetria = Path("datos/telemetria_optimizacion.csv")

    def procesar_y_generar_telemetria(self) -> Path:
        if not self.archivo_parquet.exists():
            raise FileNotFoundError(f"Archivo Parquet ausente: {self.archivo_parquet}")

        df_completo = pl.read_parquet(self.archivo_parquet)
        archivos_ordenados = df_completo["archivo_origen"].unique().sort().to_list()
        logger.info(f"Procesando {len(archivos_ordenados)} series temporales...")
        registros_telemetria = []

        for idx, nombre_archivo in enumerate(archivos_ordenados):
            t_inicio = time.perf_counter_ns()
            df_bloque = df_completo.filter(pl.col("archivo_origen") == nombre_archivo)

            rpm_datos = df_bloque["rpm"].to_numpy()
            media_rpm = np.mean(rpm_datos)
            desv_rpm = np.std(rpm_datos)
            cv_rpm = (desv_rpm / media_rpm) * 100 if media_rpm > 0 else 0
            if cv_rpm > self.umbral_cv_rpm:
                continue

            señal_cruda = df_bloque["rodamiento_1"].head(self.puntos_fft).to_numpy()
            if len(señal_cruda) < self.puntos_fft:
                continue

            frecuencias, amplitudes = ejecutar_fft(señal_cruda, self.frecuencia_muestreo)
            valor_rms = calcular_rms(señal_cruda)
            idx_pico = np.argmax(amplitudes)
            frecuencia_dom = frecuencias[idx_pico] if len(frecuencias) > 0 else 0
            amplitud_dom = amplitudes[idx_pico] if len(amplitudes) > 0 else 0

            t_fin = time.perf_counter_ns()
            tiempo_computo_ms = (t_fin - t_inicio) / 1_000_000.0

            registros_telemetria.append({
                "indice_secuencial": idx,
                "archivo_origen": nombre_archivo,
                "tiempo_computo_ms": tiempo_computo_ms,
                "cv_rpm": cv_rpm,
                "vibracion_rms": valor_rms,
                "frecuencia_dominante_hz": frecuencia_dom,
                "amplitud_maxima_g": amplitud_dom
            })

        df_telemetria = pl.DataFrame(registros_telemetria)
        df_telemetria.write_csv(self.archivo_telemetria)
        logger.info(f"Telemetria generada en: {self.archivo_telemetria}")
        return self.archivo_telemetria

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    analizador = MotorAnalisisVibraciones()
    try:
        analizador.procesar_y_generar_telemetria()
    except Exception as e:
        logger.error(f"Fallo: {e}")
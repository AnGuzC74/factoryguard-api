import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import polars as pl
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True), logging.FileHandler("datos/pipeline.log", encoding="utf-8")]
)
logger = logging.getLogger("pipeline_ingesta")

def procesar_un_archivo_nasa(ruta_archivo: Path) -> pl.DataFrame:
    try:
        marca_tiempo = ruta_archivo.name
        df = pl.read_csv(
            ruta_archivo,
            has_header=False,
            separator="\t",
            new_columns=["rodamiento_1", "rodamiento_2", "rodamiento_3", "rodamiento_4"]
        )
        df = df.with_columns([
            pl.lit(marca_tiempo).alias("archivo_origen"),
            pl.lit(2000.0).alias("rpm"),
            pl.lit(45.2).alias("temperatura")
        ])
        return df
    except Exception as e:
        logger.error(f"Error en {ruta_archivo.name}: {e}")
        return pl.DataFrame()

@dataclass
class PipelineIngestaIndustrial:
    ruta_config: Path = Path("config.toml")

    def __post_init__(self):
        with open(self.ruta_config, "rb") as f:
            self.config = tomllib.load(f)
        self.ruta_datos = Path(self.config["infraestructura"]["ruta_datos"])
        self.archivo_final = Path(self.config["infraestructura"]["archivo_parquet"])
        self.ruta_datos.mkdir(parents=True, exist_ok=True)

    def ejecutar_multiprocesamiento(self, carpeta_cruda: Path) -> None:
        archivos = [p for p in carpeta_cruda.iterdir() if p.is_file() and not p.name.startswith(".")]
        if not archivos:
            logger.error(f"No se encontraron archivos validos en {carpeta_cruda}")
            return

        logger.info(f"Iniciando ejecucion paralela para {len(archivos)} archivos...")
        temp_dir = self.ruta_datos / "temp_parquet"
        temp_dir.mkdir(exist_ok=True)

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(procesar_un_archivo_nasa, f) for f in archivos]
            for i, future in enumerate(futures):
                df = future.result()
                if df.height > 0:
                    df.write_parquet(temp_dir / f"part_{i:05d}.parquet")
                if i % 100 == 0:
                    logger.info(f"Procesados {i+1}/{len(archivos)} archivos...")

        logger.info("Consolidando archivos temporales con Polars Lazy...")
        df_final = pl.scan_parquet(str(temp_dir / "*.parquet")).collect()
        df_final.write_parquet(self.archivo_final, compression="snappy")

        for f in temp_dir.glob("*.parquet"):
            f.unlink()
        temp_dir.rmdir()

        logger.info(f"Procesamiento finalizado. Parquet guardado en: {self.archivo_final} ({df_final.height:,} filas)")

if __name__ == "__main__":
    pipeline = PipelineIngestaIndustrial()
    ruta_origen = pipeline.ruta_datos / "2nd_test"
    if ruta_origen.exists():
        pipeline.ejecutar_multiprocesamiento(ruta_origen)
    else:
        logger.warning(f"Depósito de datos crudos no encontrado en: {ruta_origen}")
"""
Módulo de monitoreo programado para múltiples activos.
Revisa el estado de todos los rodamientos/motores registrados y genera alertas
con deduplicación por ventana temporal.
"""
from pathlib import Path
import tomllib
import polars as pl
from typing import Dict, Any

from ..database.db_manager import DatabaseManager
from ..alert.alert_manager import AlertManager


class AssetMonitor:
    def __init__(self, config_path: Path = Path("config.toml")):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)
        self.db = DatabaseManager()
        self.alert = AlertManager()
        self.umbral_critico = self.config["umbrales_severidad"]["critico_rms"]
        self.umbral_alerta = self.config["umbrales_severidad"]["alerta_rms"]
        self.csv_path = Path("datos/telemetria_optimizacion.csv")

    def monitorear_todos(self) -> None:
        assets = self.db.get_assets()
        if not assets:
            print("[MONITOR] No hay activos registrados.")
            return

        print(f"[MONITOR] Revisando {len(assets)} activos...")
        alerts_generadas = 0

        for asset in assets:
            last = self.db.get_latest_measurement(asset["id"])
            if last is None:
                continue

            if last["rms_max_historico"] >= self.umbral_critico:
                tipo = "CRÍTICO"
                mensaje = f"RUL agotado en {asset['name']}. Zona: {last['zona_falla']}"
                recomendacion = "Reemplazo inmediato"
            elif last["rms_max_historico"] >= self.umbral_alerta and last["rul_hours"] < 50:
                tipo = "ALERTA"
                mensaje = f"RUL bajo ({last['rul_hours']:.1f}h) en {asset['name']}"
                recomendacion = "Programar reemplazo en próximas horas"
            else:
                continue

            if not self.db.has_recent_alert(asset["id"], tipo=tipo, horas=24):
                self.db.save_alert(asset["id"], tipo, mensaje)
                self.alert.send_alert(
                    asset["name"],
                    {
                        "rms_actual": last["rms_actual"],
                        "rms_max": last["rms_max_historico"],
                        "rul_hours": last["rul_hours"],
                        "zona_falla": last["zona_falla"],
                        "recomendacion": recomendacion
                    }
                )
                alerts_generadas += 1

        print(f"[MONITOR] Monitoreo completado. Alertas generadas: {alerts_generadas}")
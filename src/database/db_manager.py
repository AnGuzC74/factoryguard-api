"""
Gestor de Base de Datos SQLite para el sistema de pronóstico industrial.
Multi-activo, thread-safe para FastAPI, con context managers.
"""
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import polars as pl


class DatabaseManager:
    def __init__(self, db_path: str = "datos/industrial_ai.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self):
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    location TEXT,
                    rpm FLOAT DEFAULT 2000.0,
                    umbral_alerta FLOAT DEFAULT 0.12,
                    umbral_critico FLOAT DEFAULT 0.25,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_demo BOOLEAN DEFAULT 0
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    indice_secuencial INTEGER,
                    archivo_origen TEXT,
                    rms_actual FLOAT,
                    rms_max_historico FLOAT,
                    frecuencia_dominante FLOAT,
                    rul_hours FLOAT,
                    modelo_usado TEXT,
                    zona_falla TEXT,
                    FOREIGN KEY (asset_id) REFERENCES assets(id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tipo TEXT NOT NULL,
                    mensaje TEXT NOT NULL,
                    leido BOOLEAN DEFAULT 0,
                    FOREIGN KEY (asset_id) REFERENCES assets(id)
                )
            ''')
            conn.commit()

    # --- Operaciones con Activos ---

    def register_asset(self, name: str, description: str = "", location: str = "",
                       rpm: float = 2000.0, umbral_alerta: float = 0.12,
                       umbral_critico: float = 0.25, is_demo: bool = False) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO assets
                (name, description, location, rpm, umbral_alerta, umbral_critico, is_demo)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, description, location, rpm, umbral_alerta, umbral_critico, 1 if is_demo else 0))
            conn.commit()
            return cursor.lastrowid

    def get_assets(self, include_demo: bool = False) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if include_demo:
                cursor.execute('''
                    SELECT id, name, description, location, rpm,
                           umbral_alerta, umbral_critico, created_at, is_demo
                    FROM assets ORDER BY name
                ''')
            else:
                cursor.execute('''
                    SELECT id, name, description, location, rpm,
                           umbral_alerta, umbral_critico, created_at, is_demo
                    FROM assets WHERE is_demo = 0 ORDER BY name
                ''')
            return [dict(row) for row in cursor.fetchall()]

    def get_asset_by_id(self, asset_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, description, location, rpm,
                       umbral_alerta, umbral_critico, created_at, is_demo
                FROM assets WHERE id = ?
            ''', (asset_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_asset_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, description, location, rpm,
                       umbral_alerta, umbral_critico, created_at, is_demo
                FROM assets WHERE name = ?
            ''', (name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # --- Operaciones con Mediciones ---

    def save_measurement(self, asset_id: int, indice_secuencial: int, archivo_origen: str,
                         rms_actual: float, rms_max_historico: float,
                         frecuencia_dominante: float, rul_hours: float,
                         modelo_usado: str, zona_falla: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO measurements
                (asset_id, indice_secuencial, archivo_origen, rms_actual, rms_max_historico,
                 frecuencia_dominante, rul_hours, modelo_usado, zona_falla)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (asset_id, indice_secuencial, archivo_origen, rms_actual, rms_max_historico,
                  frecuencia_dominante, rul_hours, modelo_usado, zona_falla))
            conn.commit()

    def get_latest_measurement(self, asset_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, indice_secuencial, archivo_origen,
                       rms_actual, rms_max_historico, frecuencia_dominante,
                       rul_hours, modelo_usado, zona_falla
                FROM measurements
                WHERE asset_id = ?
                ORDER BY indice_secuencial DESC
                LIMIT 1
            ''', (asset_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_measurements(self, asset_id: int, limit: int = 10000) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, indice_secuencial, archivo_origen,
                       rms_actual, rms_max_historico, frecuencia_dominante,
                       rul_hours, modelo_usado, zona_falla
                FROM measurements
                WHERE asset_id = ?
                ORDER BY indice_secuencial ASC
                LIMIT ?
            ''', (asset_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    # --- Operaciones con Alertas ---

    def save_alert(self, asset_id: int, tipo: str, mensaje: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (asset_id, tipo, mensaje)
                VALUES (?, ?, ?)
            ''', (asset_id, tipo, mensaje))
            conn.commit()

    def get_alerts(self, asset_id: Optional[int] = None, leido: bool = False) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if asset_id is None:
                cursor.execute('''
                    SELECT id, asset_id, timestamp, tipo, mensaje, leido
                    FROM alerts
                    WHERE leido = ?
                    ORDER BY timestamp DESC
                ''', (1 if leido else 0,))
            else:
                cursor.execute('''
                    SELECT id, asset_id, timestamp, tipo, mensaje, leido
                    FROM alerts
                    WHERE asset_id = ? AND leido = ?
                    ORDER BY timestamp DESC
                ''', (asset_id, 1 if leido else 0))
            return [dict(row) for row in cursor.fetchall()]

    def mark_alert_read(self, alert_id: int) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE alerts SET leido = 1 WHERE id = ?', (alert_id,))
            conn.commit()

    def has_recent_alert(self, asset_id: int, tipo: str, horas: float = 24.0) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM alerts
                WHERE asset_id = ?
                  AND tipo = ?
                  AND timestamp >= datetime('now', ?)
            ''', (asset_id, tipo, f'-{horas} hours'))
            count = cursor.fetchone()[0]
            return count > 0

    # --- Eliminación ---

    def delete_asset(self, asset_id: int) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alerts WHERE asset_id = ?", (asset_id,))
            cursor.execute("DELETE FROM measurements WHERE asset_id = ?", (asset_id,))
            cursor.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
            conn.commit()

    # --- Utilidades ---

    def import_from_csv(self, csv_path: str, asset_name: str = "Rodamiento NASA") -> int:
        df = pl.read_csv(csv_path)
        asset = self.get_asset_by_name(asset_name)
        if asset is None:
            asset_id = self.register_asset(name=asset_name, description="Importado desde CSV")
        else:
            asset_id = asset["id"]

        batch = []
        rms_acumulado_max = 0.0

        for row in df.rows():
            try:
                idx = int(row[df.columns.index("indice_secuencial")])
                archivo = str(row[df.columns.index("archivo_origen")])
                rms = float(row[df.columns.index("vibracion_rms")])
                freq = float(row[df.columns.index("frecuencia_dominante_hz")])
                rms_acumulado_max = max(rms_acumulado_max, rms)
                batch.append((
                    asset_id,
                    idx,
                    archivo,
                    rms,
                    rms_acumulado_max,
                    freq,
                    0.0,
                    "Importado",
                    "No definida"
                ))
            except Exception as e:
                print(f"Error importando fila: {e}")
                continue

        if batch:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany('''
                    INSERT INTO measurements
                    (asset_id, indice_secuencial, archivo_origen, rms_actual,
                     rms_max_historico, frecuencia_dominante, rul_hours,
                     modelo_usado, zona_falla)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch)
                conn.commit()

        return asset_id
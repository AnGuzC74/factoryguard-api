#!/usr/bin/env python3
"""
Genera un reporte de diagnóstico en la terminal usando Rich.
Incluye timestamp, RUL calculado y estado detallado.
"""
import sys
import tomllib
from pathlib import Path
from datetime import datetime
import polars as pl
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.dsp import calcular_rul_hibrido

console = Console()

def generar_reporte_terminal(archivo_origen: str = None):
    config_path = Path("config.toml")
    if not config_path.exists():
        console.print("[red]Error: config.toml no encontrado[/red]")
        return

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    csv_path = Path("datos/telemetria_optimizacion.csv")
    if not csv_path.exists():
        console.print("[red]Error: telemetria_optimizacion.csv no encontrado[/red]")
        return

    df = pl.read_csv(csv_path)
    umbral_critico = config["umbrales_severidad"]["critico_rms"]
    umbral_alerta = config["umbrales_severidad"]["alerta_rms"]
    minutos_por_ciclo = config["diagnostico"]["minutos_por_ciclo"]

    if archivo_origen is None:
        datos = df[-1].to_dicts()[0]
        archivo_origen = datos["archivo_origen"]
    else:
        fila = df.filter(pl.col("archivo_origen") == archivo_origen)
        if fila.is_empty():
            console.print(f"[red]Archivo {archivo_origen} no encontrado[/red]")
            return
        datos = fila.to_dicts()[0]

    idx_actual = datos["indice_secuencial"]
    rms_actual = datos["vibracion_rms"]
    freq_dom = datos["frecuencia_dominante_hz"]

    df_hist = df.filter(pl.col("indice_secuencial") <= idx_actual)
    max_rms_historico = df_hist["vibracion_rms"].max()

    x_ciclos = df_hist["indice_secuencial"].to_numpy()
    y_rms = df_hist["vibracion_rms"].to_numpy()
    ciclos_rul, _, _, modelo = calcular_rul_hibrido(x_ciclos, y_rms, umbral_critico)
    horas_rul = (ciclos_rul * minutos_por_ciclo) / 60.0

    if max_rms_historico >= umbral_critico:
        estado = "🔴 CRÍTICO - Reemplazo inmediato"
        color = "red"
    elif max_rms_historico >= umbral_alerta:
        if ciclos_rul < 50:
            estado = "🟠 ALERTA AVANZADA"
            color = "orange1"
        elif ciclos_rul < 200:
            estado = "🟡 ALERTA INCIPIENTE"
            color = "yellow"
        else:
            estado = "🟣 VIGILANCIA"
            color = "magenta"
    else:
        estado = "🟢 NORMAL - Operación segura"
        color = "green"

    table = Table(title=f"Diagnóstico para {archivo_origen}", box=box.ROUNDED)
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="magenta")
    table.add_row("RMS Actual", f"{rms_actual:.4f} g")
    table.add_row("RMS Máximo Histórico", f"{max_rms_historico:.4f} g")
    table.add_row("Frecuencia Dominante", f"{freq_dom:.1f} Hz")
    rul_text = f"{horas_rul:.1f} horas" if ciclos_rul < 999 else "Estable / Sin pronóstico"
    table.add_row("RUL Estimado", rul_text)
    table.add_row("Modelo usado", modelo)
    table.add_row("Estado", f"[{color}]{estado}[/{color}]")

    panel = Panel(
        table,
        title=f"🏭 Pronóstico Industrial - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="blue"
    )
    console.print(panel)

if __name__ == "__main__":
    import sys
    archivo = sys.argv[1] if len(sys.argv) > 1 else None
    generar_reporte_terminal(archivo)
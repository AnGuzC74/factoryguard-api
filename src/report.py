#!/usr/bin/env python3
"""
Genera un reporte de diagnóstico en la terminal usando Rich.
"""
import tomllib
from pathlib import Path
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

def generar_reporte():
    # Cargar configuración y datos
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    df = pl.read_csv("datos/telemetria_optimizacion.csv")
    ultimo = df[-1]
    max_rms = df["vibracion_rms"].max()
    
    # Determinar estado (reglas SIMPLES y CLARAS)
    umbral_critico = config["umbrales_severidad"]["critico_rms"]
    umbral_alerta = config["umbrales_severidad"]["alerta_rms"]
    
    if max_rms >= umbral_critico:
        estado = "🔴 CRÍTICO - Reemplazo inmediato"
    elif max_rms >= umbral_alerta:
        estado = "🟡 ALERTA - Programar mantenimiento"
    else:
        estado = "🟢 NORMAL - Operación segura"
    
    # Tabla
    table = Table(title="Diagnóstico de Rodamiento")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="magenta")
    table.add_row("RMS Actual", f"{ultimo['vibracion_rms']:.4f} g")
    table.add_row("RMS Máximo Histórico", f"{max_rms:.4f} g")
    table.add_row("Frecuencia Dominante", f"{ultimo['frecuencia_dominante_hz']:.1f} Hz")
    table.add_row("Estado", estado)
    
    console.print(Panel(table, title="🏭 Pronóstico Industrial", border_style="blue"))

if __name__ == "__main__":
    generar_reporte()
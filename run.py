#!/usr/bin/env python3
"""
Orquestador Inteligente con Menú Interactivo para el Sistema de Pronóstico Industrial.
Soporta multi-activo, alertas, RAG, monitoreo automático, gestión de equipos y demostración guiada.
"""
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import webbrowser
from typing import Optional, List, Dict, Any

import numpy as np
import tomllib
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

from src.database.db_manager import DatabaseManager
from src.alert.alert_manager import AlertManager
from src.core.dsp import calcular_rul_hibrido

console = Console()

ROOT = Path(__file__).parent
DATOS_DIR = ROOT / "datos"
PARQUET_PATH = DATOS_DIR / "nasa_bearing_consolidado.parquet"
CSV_PATH = DATOS_DIR / "telemetria_optimizacion.csv"

db = DatabaseManager()
alert_mgr = AlertManager()

# Cargar configuración
with open(ROOT / "config.toml", "rb") as f:
    CONFIG = tomllib.load(f)
UMBRAL_CRITICO = CONFIG["umbrales_severidad"]["critico_rms"]
UMBRAL_ALERTA = CONFIG["umbrales_severidad"]["alerta_rms"]
MINUTOS_POR_CICLO = CONFIG["diagnostico"]["minutos_por_ciclo"]
TOLERANCIA_ARMONICA = CONFIG["diagnostico"]["tolerancia_armonica_hz"]

# Frecuencias teóricas
FREQ_BPFO = 236.4
FREQ_BPFI = 296.8
FREQ_BSF = 139.2
FREQ_FTF = 14.8


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def verificar_datos() -> None:
    if not PARQUET_PATH.exists():
        console.print("[yellow]⚠️ Archivo Parquet no encontrado. Ejecutando ingesta...[/yellow]")
        ejecutar_script(ROOT / "src" / "pipeline" / "ingesta.py")
    if not CSV_PATH.exists():
        console.print("[yellow]⚠️ Archivo de telemetría no encontrado. Ejecutando análisis...[/yellow]")
        ejecutar_script(ROOT / "src" / "pipeline" / "analisis.py")
    else:
        if PARQUET_PATH.stat().st_mtime > CSV_PATH.stat().st_mtime:
            console.print("[yellow]⚠️ El Parquet es más reciente que el CSV. Actualizando telemetría...[/yellow]")
            ejecutar_script(ROOT / "src" / "pipeline" / "analisis.py")


def ejecutar_script(script_path: Path) -> bool:
    try:
        console.print(f"[cyan]▶ Ejecutando: {script_path.name}[/cyan]")
        resultado = subprocess.run([sys.executable, str(script_path)], capture_output=False, check=False)
        if resultado.returncode == 0:
            console.print("[green]✅ Completado[/green]")
            return True
        else:
            console.print(f"[red]❌ Falló con código {resultado.returncode}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return False


def obtener_lista_archivos_csv() -> List[str]:
    if not CSV_PATH.exists():
        return []
    import polars as pl
    df = pl.read_csv(CSV_PATH)
    return df["archivo_origen"].to_list()


# ============================================================
# FUNCIONES DE DIAGNÓSTICO EN VIVO
# ============================================================

def verificar_armonicos(freq: float, base: float, max_orden: int = 5) -> bool:
    for orden in range(1, max_orden + 1):
        if abs(freq - (base * orden)) <= TOLERANCIA_ARMONICA:
            return True
    return False


def calcular_zona_falla(freq_dom: float, max_rms: float) -> str:
    if max_rms < UMBRAL_ALERTA:
        return "Ninguna (operación normal)"
    if verificar_armonicos(freq_dom, FREQ_BPFO) or freq_dom > 3500:
        return "Pista Externa (BPFO)"
    elif verificar_armonicos(freq_dom, FREQ_BPFI):
        return "Pista Interna (BPFI)"
    elif verificar_armonicos(freq_dom, FREQ_BSF):
        return "Elementos Rodantes (BSF)"
    elif verificar_armonicos(freq_dom, FREQ_FTF):
        return "Jaula (FTF)"
    else:
        return "Pista Externa (Modulación)"


def recalcular_rul_en_vivo(asset_id: int) -> Optional[Dict[str, Any]]:
    measurements = db.get_measurements(asset_id, limit=10000)
    if not measurements or len(measurements) < 2:
        return None

    x = np.array([m["indice_secuencial"] for m in measurements])
    y = np.array([m["rms_actual"] for m in measurements])
    freq_dom = measurements[-1].get("frecuencia_dominante", 0)

    ciclos, _, _, modelo = calcular_rul_hibrido(x, y, UMBRAL_CRITICO, horizonte_prediccion=300)
    horas = (ciclos * MINUTOS_POR_CICLO) / 60.0 if ciclos < 999 else None
    dias = horas / 24 if horas is not None else None

    max_rms = float(np.max(y))
    zona = calcular_zona_falla(freq_dom, max_rms)

    if max_rms >= UMBRAL_CRITICO:
        estado = "CRÍTICO (Reemplazo)"
    elif max_rms >= UMBRAL_ALERTA:
        if horas is not None and horas < 8:
            estado = "ALERTA AVANZADA"
        elif horas is not None and horas < 50:
            estado = "ALERTA INCIPIENTE"
        else:
            estado = "VIGILANCIA"
    else:
        estado = "SALUDABLE"

    return {
        "rms_actual": float(y[-1]),
        "max_rms": max_rms,
        "frecuencia_dominante": freq_dom,
        "rul_hours": horas,
        "rul_days": dias,
        "ciclos": ciclos,
        "modelo": modelo,
        "zona_falla": zona,
        "estado": estado,
        "ultima_fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ============================================================
# DEMOSTRACIÓN GUIADA (CON RAG)
# ============================================================

def opcion_demo_guiada() -> None:
    console.clear()
    console.print(Panel.fit(
        "🚀 [bold]DEMOSTRACIÓN GUIADA - NASA IMS[/bold]",
        border_style="cyan"
    ))

    console.print("\n[bold]Este ejemplo te guiará a través de la evolución de un rodamiento[/bold]")
    console.print("desde su estado sano hasta la falla crítica, usando el dataset NASA IMS.\n")

    if not Confirm.ask("¿Quieres ver la evolución completa en 4 pasos?"):
        return

    # Registrar activo demo si no existe
    asset = db.get_asset_by_name("NASA_Rodamiento_Demo")
    if asset is None:
        asset_id = db.register_asset(
            name="NASA_Rodamiento_Demo",
            description="Rodamiento del dataset NASA IMS (Test 2) - DEMO",
            location="Laboratorio NASA",
            rpm=2000.0,
            is_demo=True
        )
        db.import_from_csv(str(CSV_PATH), "NASA_Rodamiento_Demo")
        console.print("[green]✅ Activo demo creado e importado.[/green]")
    else:
        asset_id = asset["id"]

    def mostrar_estado_ciclo(ciclo, titulo):
        measurements = db.get_measurements(asset_id, limit=10000)
        if not measurements:
            console.print("[red]No hay mediciones disponibles.[/red]")
            return
        target = None
        for m in measurements:
            if m["indice_secuencial"] <= ciclo:
                target = m
            else:
                break
        if target is None:
            target = measurements[0]
        x = np.array([m["indice_secuencial"] for m in measurements if m["indice_secuencial"] <= ciclo])
        y = np.array([m["rms_actual"] for m in measurements if m["indice_secuencial"] <= ciclo])
        if len(x) < 2:
            console.print("[yellow]Datos insuficientes para este ciclo.[/yellow]")
            return
        ciclos, _, _, modelo = calcular_rul_hibrido(x, y, UMBRAL_CRITICO)
        horas = (ciclos * MINUTOS_POR_CICLO) / 60.0 if ciclos < 999 else None
        max_rms = float(np.max(y))
        zona = calcular_zona_falla(target["frecuencia_dominante"], max_rms)

        table = Table(title=f"Estado en ciclo {ciclo} - {titulo}", box=box.ROUNDED)
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="magenta")
        table.add_row("RMS Actual", f"{target['rms_actual']:.4f} g")
        table.add_row("RMS Máximo", f"{max_rms:.4f} g")
        table.add_row("Frecuencia", f"{target['frecuencia_dominante']:.1f} Hz")
        rul_text = f"{horas:.1f} horas" if horas else "Estable / Sin proyección"
        table.add_row("RUL", rul_text)
        table.add_row("Zona de Falla", zona)
        table.add_row("Modelo", modelo)
        console.print(table)

    # PASO 1: Sano
    console.print("\n[bold green]▶ PASO 1: Estado Sano (ciclo ~100)[/bold green]")
    mostrar_estado_ciclo(100, "Sano")
    console.print("[dim]El rodamiento opera dentro de parámetros normales.[/dim]")
    Prompt.ask("\nPresiona Enter para continuar...")

    # PASO 2: Incipiente
    console.print("\n[bold yellow]▶ PASO 2: Desgaste Incipiente (ciclo ~400)[/bold yellow]")
    mostrar_estado_ciclo(400, "Incipiente")
    console.print("[dim]Se detecta un ligero incremento en la energía RMS.[/dim]")
    Prompt.ask("\nPresiona Enter para continuar...")

    # PASO 3: Alerta Avanzada
    console.print("\n[bold orange1]▶ PASO 3: Alerta Avanzada (ciclo ~700)[/bold orange1]")
    mostrar_estado_ciclo(700, "Alerta Avanzada")
    console.print("[dim]El desgaste es significativo. RUL estimado en horas.[/dim]")
    Prompt.ask("\nPresiona Enter para continuar...")

    # PASO 4: Crítico
    console.print("\n[bold red]▶ PASO 4: Falla Crítica (ciclo ~983)[/bold red]")
    mostrar_estado_ciclo(983, "Crítico")
    console.print("[dim]El rodamiento ha alcanzado el umbral crítico. Reemplazo inmediato.[/dim]")

    console.print("\n[bold green]✅ Demostración completada.[/bold green]")
    
    # Preguntar por PDF
    if Confirm.ask("¿Quieres generar un PDF de este ejemplo?"):
        try:
            from src.report.pdf_generator import generar_reporte_pdf
            archivos = obtener_lista_archivos_csv()
            if archivos:
                pdf_path = generar_reporte_pdf(archivos[-1])
                console.print(f"[green]✅ PDF generado: {pdf_path}[/green]")
                if Confirm.ask("¿Quieres abrir el PDF ahora?"):
                    webbrowser.open(str(pdf_path))
            else:
                console.print("[yellow]No hay archivos para generar PDF.[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error generando PDF: {e}[/red]")

    # --- CONSULTA RAG EN LA DEMO (NUEVO) ---
    console.print("\n[bold cyan]🧠 ¿Quieres ver una consulta RAG con el estado actual?[/bold cyan]")
    if Confirm.ask("Ejecutar agente RAG con el estado actual del rodamiento"):
        try:
            from src.agent.rag_agent import RAGAgent
            agent = RAGAgent()
            diag = recalcular_rul_en_vivo(asset_id)
            if diag:
                status = {
                    "rms_actual": diag["rms_actual"],
                    "rms_max": diag["max_rms"],
                    "frecuencia": diag["frecuencia_dominante"],
                    "rul_hours": diag["rul_hours"] if diag["rul_hours"] else 999,
                    "zona_falla": diag["zona_falla"]
                }
                console.print("[cyan]Consultando base de conocimiento...[/cyan]")
                recomendacion = agent.generar_recomendacion("NASA_Rodamiento_Demo", status)
                console.print(Panel(recomendacion, title="🤖 Recomendación para el rodamiento demostración", border_style="green"))
            else:
                console.print("[yellow]No hay suficientes datos para RAG.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error en RAG: {e}[/red]")

    console.print("\n[bold green]✅ Demostración finalizada.[/bold green]")


# ============================================================
# OPCIONES DEL MENÚ
# ============================================================

def opcion_ejemplo_nasa() -> None:
    """Carga el rodamiento de la NASA como ejemplo y registra un activo demo."""
    console.print("\n[bold cyan]🚀 Cargando ejemplo NASA IMS...[/bold cyan]")

    if not CSV_PATH.exists():
        console.print("[red]❌ No se encontró telemetria_optimizacion.csv. Ejecuta primero el pipeline.[/red]")
        return

    asset = db.get_asset_by_name("NASA_Rodamiento_Demo")
    if asset:
        console.print(f"[yellow]⚠️ El activo 'NASA_Rodamiento_Demo' ya existe (ID: {asset['id']}).[/yellow]")
        if not Confirm.ask("¿Reemplazar? (se eliminarán los datos antiguos)"):
            return
        db.delete_asset(asset["id"])

    asset_id = db.register_asset(
        name="NASA_Rodamiento_Demo",
        description="Rodamiento del dataset NASA IMS (Test 2) - DEMO",
        location="Laboratorio NASA",
        rpm=2000.0,
        is_demo=True
    )
    console.print(f"[green]✅ Activo 'NASA_Rodamiento_Demo' registrado con ID: {asset_id}[/green]")

    try:
        db.import_from_csv(str(CSV_PATH), "NASA_Rodamiento_Demo")
        console.print("[green]✅ Datos del rodamiento NASA importados exitosamente.[/green]")
        console.print("[dim]   Puedes consultar el estado con la opción 1.[/dim]")
        console.print("[dim]   Puedes ver el dashboard con la opción 3.[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Error importando datos: {e}[/red]")


def opcion_registrar_activo() -> None:
    console.print("\n[bold cyan]Registro de Nuevo Activo[/bold cyan]")
    nombre = Prompt.ask("Nombre del activo (ej. Motor1-RodamientoA)")
    descripcion = Prompt.ask("Descripción (opcional)", default="")
    ubicacion = Prompt.ask("Ubicación (opcional)", default="Planta principal")
    rpm = float(Prompt.ask("RPM", default="2000"))
    asset_id = db.register_asset(name=nombre, description=descripcion, location=ubicacion, rpm=rpm)
    console.print(f"[green]✅ Activo registrado con ID: {asset_id}[/green]")
    if Confirm.ask("¿Importar datos desde CSV de telemetría?"):
        try:
            db.import_from_csv(str(CSV_PATH), nombre)
            console.print("[green]✅ Datos importados exitosamente.[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error importando datos: {e}[/red]")


def opcion_ver_activos() -> None:
    assets = db.get_assets(include_demo=False)
    if not assets:
        console.print("[yellow]No hay activos registrados (excluyendo demo).[/yellow]")
        return
    table = Table(title="Activos Registrados", box=box.ROUNDED)
    table.add_column("ID", style="cyan")
    table.add_column("Nombre", style="white")
    table.add_column("Ubicación", style="dim")
    table.add_column("RPM", style="magenta")
    table.add_column("Estado", style="green")
    for asset in assets:
        diag = recalcular_rul_en_vivo(asset["id"])
        if diag:
            estado = diag["estado"]
            rul = f"{diag['rul_hours']:.1f}h" if diag['rul_hours'] else "Estable"
        else:
            estado = "Sin datos"
            rul = "N/A"
        table.add_row(
            str(asset["id"]),
            asset["name"],
            asset.get("location", ""),
            str(asset.get("rpm", 2000)),
            f"{estado} ({rul})"
        )
    console.print(table)


def opcion_consultar_estado() -> None:
    assets = db.get_assets(include_demo=False)
    if not assets:
        console.print("[yellow]No hay activos registrados (excluyendo demo). Registra uno primero.[/yellow]")
        return
    console.print("\n[bold cyan]Activos disponibles:[/bold cyan]")
    for i, asset in enumerate(assets, 1):
        console.print(f"  [{i}] {asset['name']} (ID: {asset['id']})")
    opcion = Prompt.ask("Selecciona un activo (número o nombre)")
    try:
        idx = int(opcion) - 1
        if 0 <= idx < len(assets):
            asset = assets[idx]
        else:
            asset = db.get_asset_by_name(opcion)
            if asset is None:
                console.print(f"[red]Activo '{opcion}' no encontrado.[/red]")
                return
    except ValueError:
        asset = db.get_asset_by_name(opcion)
        if asset is None:
            console.print(f"[red]Activo '{opcion}' no encontrado.[/red]")
            return

    diag = recalcular_rul_en_vivo(asset["id"])
    if not diag:
        console.print("[yellow]No hay suficientes datos para generar diagnóstico.[/yellow]")
        return

    table = Table(title=f"Estado de {asset['name']}", box=box.ROUNDED)
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="magenta")
    table.add_row("Fecha", diag["ultima_fecha"])
    table.add_row("RMS Actual", f"{diag['rms_actual']:.4f} g")
    table.add_row("RMS Máximo", f"{diag['max_rms']:.4f} g")
    table.add_row("Frecuencia", f"{diag['frecuencia_dominante']:.1f} Hz")
    rul_text = f"{diag['rul_hours']:.1f} horas" if diag['rul_hours'] is not None else "Estable / Sin proyección"
    if diag['rul_days'] is not None:
        rul_text += f" ({diag['rul_days']:.1f} días)"
    table.add_row("RUL", rul_text)
    table.add_row("Zona de Falla", diag['zona_falla'])
    table.add_row("Modelo", diag['modelo'])
    table.add_row("Estado", diag['estado'])
    console.print(table)


def opcion_eliminar_activo() -> None:
    assets = db.get_assets(include_demo=False)
    if not assets:
        console.print("[yellow]No hay activos registrados (excluyendo demo).[/yellow]")
        return
    console.print("\n[bold cyan]Selecciona un activo para eliminar:[/bold cyan]")
    for i, asset in enumerate(assets, 1):
        console.print(f"  [{i}] {asset['name']} (ID: {asset['id']})")
    opcion = Prompt.ask("Número (0 para cancelar)")
    try:
        idx = int(opcion) - 1
        if idx < 0:
            return
        asset = assets[idx]
    except:
        console.print("[red]Selección inválida.[/red]")
        return
    if Confirm.ask(f"¿Eliminar el activo '{asset['name']}' y TODOS sus datos?"):
        db.delete_asset(asset["id"])
        console.print(f"[green]✅ Activo '{asset['name']}' eliminado.[/green]")


def opcion_alertas_pendientes() -> None:
    alerts = db.get_alerts(leido=False)
    if not alerts:
        console.print("[green]✅ No hay alertas pendientes.[/green]")
        return
    table = Table(title="Alertas Pendientes", box=box.ROUNDED)
    table.add_column("ID", style="cyan")
    table.add_column("Activo", style="white")
    table.add_column("Tipo", style="yellow")
    table.add_column("Fecha", style="dim")
    table.add_column("Mensaje", style="white")
    for alert in alerts:
        asset = db.get_asset_by_id(alert["asset_id"])
        nombre = asset["name"] if asset else f"ID {alert['asset_id']}"
        table.add_row(
            str(alert["id"]),
            nombre,
            alert["tipo"],
            alert["timestamp"][:16],
            alert["mensaje"][:40] + "..." if len(alert["mensaje"]) > 40 else alert["mensaje"]
        )
    console.print(table)
    if Confirm.ask("¿Marcar todas las alertas como leídas?"):
        for alert in alerts:
            db.mark_alert_read(alert["id"])
        console.print("[green]✅ Alertas marcadas como leídas.[/green]")


def opcion_generar_pdf() -> None:
    archivos = obtener_lista_archivos_csv()
    if not archivos:
        console.print("[red]No hay archivos de telemetría disponibles.[/red]")
        return
    console.print("\n[bold cyan]Selecciona un archivo para generar el PDF:[/bold cyan]")
    for i, arch in enumerate(archivos[-10:], 1):
        console.print(f"  [{i}] {arch}")
    console.print("  [0] Último archivo")
    console.print("  [m] Escribir nombre de archivo manualmente")
    opcion = Prompt.ask("Ingresa el número o 'm'", default="0")
    if opcion.lower() == 'm':
        archivo_seleccionado = Prompt.ask("Escribe el nombre completo del archivo")
    else:
        idx = int(opcion)
        archivo_seleccionado = archivos[-1] if idx == 0 else archivos[-10:][idx-1]
    try:
        sys.path.insert(0, str(ROOT))
        from src.report.pdf_generator import generar_reporte_pdf
        pdf_path = generar_reporte_pdf(archivo_seleccionado)
        console.print(f"[green]✅ PDF generado: {pdf_path}[/green]")
        if Confirm.ask("¿Quieres abrir el PDF ahora?"):
            webbrowser.open(str(pdf_path))
    except Exception as e:
        console.print(f"[red]❌ Error al generar PDF: {e}[/red]")


def opcion_listar_csvs() -> None:
    archivos = obtener_lista_archivos_csv()
    if not archivos:
        console.print("[red]No hay archivos de telemetría disponibles.[/red]")
        return
    table = Table(title="📂 Archivos de Telemetría Disponibles", box=box.ROUNDED)
    table.add_column("#", style="cyan")
    table.add_column("Nombre del archivo", style="white")
    for i, arch in enumerate(archivos, 1):
        table.add_row(str(i), arch)
    console.print(table)
    console.print(f"[dim]Total: {len(archivos)} archivos[/dim]")
    console.print("[dim]💡 Estos archivos pueden ser importados a un activo usando la opción 4.[/dim]")


def opcion_agente_rag() -> None:
    assets = db.get_assets(include_demo=False)
    if not assets:
        console.print("[yellow]No hay activos registrados (excluyendo demo).[/yellow]")
        return
    console.print("\n[bold cyan]Selecciona un activo para analizar:[/bold cyan]")
    for i, asset in enumerate(assets, 1):
        console.print(f"  [{i}] {asset['name']}")
    opcion = Prompt.ask("Número", default="1")
    try:
        idx = int(opcion) - 1
        if idx < 0 or idx >= len(assets):
            console.print("[red]Selección inválida.[/red]")
            return
        asset = assets[idx]
    except:
        console.print("[red]Selección inválida.[/red]")
        return

    diag = recalcular_rul_en_vivo(asset["id"])
    if not diag:
        console.print("[yellow]No hay suficientes datos para el agente RAG.[/yellow]")
        return

    try:
        from src.agent.rag_agent import RAGAgent
        agent = RAGAgent()
        status = {
            "rms_actual": diag["rms_actual"],
            "rms_max": diag["max_rms"],
            "frecuencia": diag["frecuencia_dominante"],
            "rul_hours": diag["rul_hours"] if diag["rul_hours"] else 999,
            "zona_falla": diag["zona_falla"]
        }
        console.print("[cyan]Consultando base de conocimiento...[/cyan]")
        recomendacion = agent.generar_recomendacion(asset['name'], status)
        console.print(Panel(recomendacion, title=f"🤖 Recomendación para {asset['name']}", border_style="green"))
    except ImportError as e:
        console.print(f"[red]Error: No se pudo importar RAGAgent. Verifica que src/agent/rag_agent.py existe.[/red]")
        console.print(f"[dim]Detalle: {e}[/dim]")
    except Exception as e:
        console.print(f"[red]Error ejecutando agente RAG: {e}[/red]")


def opcion_dashboard() -> None:
    console.print("[cyan]▶ Lanzando dashboard en segundo plano...[/cyan]")
    webbrowser.open("http://localhost:8501")
    try:
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            str(ROOT / "src" / "app" / "dashboard.py"),
            "--server.address=localhost",
            "--server.port=8501"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        console.print("[green]✅ Dashboard iniciado en http://localhost:8501[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


def opcion_monitoreo() -> None:
    console.print("[cyan]▶ Ejecutando monitoreo automático de todos los activos...[/cyan]")
    try:
        from src.monitor.monitor import AssetMonitor
        monitor = AssetMonitor()
        monitor.monitorear_todos()
        console.print("[green]✅ Monitoreo completado.[/green]")
    except ImportError as e:
        console.print(f"[red]Error: No se pudo importar AssetMonitor. Verifica que src/monitor/monitor.py existe.[/red]")
        console.print(f"[dim]Detalle: {e}[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Error en monitoreo: {e}[/red]")


def opcion_info() -> None:
    console.print("\n[bold]ℹ️ Información del Sistema[/bold]")
    console.print(f"  📁 Directorio del proyecto: {ROOT}")
    if PARQUET_PATH.exists():
        size_mb = PARQUET_PATH.stat().st_size / (1024 * 1024)
        console.print(f"  📊 Parquet: {PARQUET_PATH.name} ({size_mb:.2f} MB)")
    else:
        console.print("  📊 Parquet: [red]No encontrado[/red]")
    if CSV_PATH.exists():
        import polars as pl
        df = pl.read_csv(CSV_PATH)
        console.print(f"  📈 Telemetría: {CSV_PATH.name} ({df.height} registros)")
    else:
        console.print("  📈 Telemetría: [red]No encontrada[/red]")
    assets = db.get_assets(include_demo=True)
    console.print(f"  🗃️  Activos en DB: {len(assets)}")
    try:
        import chromadb
        client = chromadb.PersistentClient(path="datos/chroma_db")
        collections = client.list_collections()
        console.print(f"  📚 Colecciones ChromaDB: {len(collections)}")
    except:
        console.print("  📚 ChromaDB: [yellow]No inicializada[/yellow]")
    Prompt.ask("\nPresiona Enter para continuar...")


def opcion_ayuda() -> None:
    console.clear()
    console.print(Panel.fit("📖 [bold]GUÍA RÁPIDA - SISTEMA DE IA INDUSTRIAL[/bold]", border_style="cyan"))
    table = Table(show_header=True, box=box.ROUNDED)
    table.add_column("Opción", style="cyan", width=8)
    table.add_column("Descripción", style="white")
    table.add_column("¿Para qué sirve?", style="dim")
    table.add_row("1", "📊 Consultar estado de activo", "Muestra RUL, frecuencia, zona de falla (recalculado en vivo).")
    table.add_row("2", "📄 Generar reporte PDF", "Informe ejecutivo con gráficos y diagnóstico.")
    table.add_row("3", "🚀 Abrir Dashboard", "Interfaz web interactiva (Streamlit).")
    table.add_row("4", "🔧 Registrar nuevo activo", "Añade un rodamiento/motor al sistema.")
    table.add_row("5", "📋 Ver activos registrados", "Lista todos los activos con su estado resumido.")
    table.add_row("6", "🔔 Ver alertas pendientes", "Muestra notificaciones activas.")
    table.add_row("7", "🧠 Agente RAG", "Recomendación experta basada en manuales (ChromaDB).")
    table.add_row("8", "ℹ️  Información del sistema", "Rutas, tamaños, activos, ChromaDB.")
    table.add_row("9", "🔄 Monitoreo automático", "Revisa todos los activos y genera alertas.")
    table.add_row("E", "🗑️ Eliminar activo", "Elimina un activo y todos sus datos.")
    table.add_row("L", "📂 Listar archivos CSV", "Muestra los archivos de telemetría disponibles (global).")
    table.add_row("D", "🚀 Demostración guiada", "Recorrido paso a paso del ejemplo NASA con RAG incluido.")
    table.add_row("N", "📥 Cargar ejemplo NASA", "Registra el rodamiento NASA como demo y carga todos los datos.")
    table.add_row("?", "📖 Ayuda", "Esta guía.")
    table.add_row("0", "❌ Salir", "Cierra el orquestador.")
    console.print(table)
    console.print("\n[dim]💡 Los diagnósticos se calculan en tiempo real usando el historial completo de mediciones.[/dim]")
    Prompt.ask("\nPresiona Enter para volver al menú...")


# ============================================================
# MENÚ PRINCIPAL
# ============================================================

def mostrar_menu() -> str:
    console.clear()
    console.print(Panel.fit("🏭 [bold]Sistema de IA Industrial - Orquestador[/bold]", border_style="blue"))

    table = Table(show_header=False, box=box.ROUNDED)
    table.add_column("Opción", style="cyan", width=4)
    table.add_column("Descripción", style="white")

    table.add_row("1", "📊 Consultar estado de activo")
    table.add_row("2", "📄 Generar reporte PDF")
    table.add_row("3", "🚀 Abrir Dashboard (Streamlit)")
    table.add_row("4", "🔧 Registrar nuevo activo")
    table.add_row("5", "📋 Ver activos registrados")
    table.add_row("6", "🔔 Ver alertas pendientes")
    table.add_row("7", "🧠 Consultar agente RAG")
    table.add_row("8", "ℹ️  Información del sistema")
    table.add_row("9", "🔄 Ejecutar monitoreo automático")
    table.add_row("E", "🗑️ Eliminar activo")
    table.add_row("L", "📂 Listar archivos CSV")
    table.add_row("D", "🚀 Demostración guiada (ejemplo NASA)")
    table.add_row("N", "📥 Cargar ejemplo NASA")
    table.add_row("?", "📖 Ayuda / Guía rápida")
    table.add_row("0", "❌ Salir")

    console.print(table)
    console.print("[dim]Selecciona una opción (o '?' para ayuda):[/dim]")
    opcion = Prompt.ask("", choices=["0","1","2","3","4","5","6","7","8","9","E","L","D","N","?"], default="0")
    return opcion.upper()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    verificar_datos()

    while True:
        opcion = mostrar_menu()

        if opcion == "0":
            console.print("[green]👋 ¡Hasta luego![/green]")
            break
        elif opcion == "?":
            opcion_ayuda()
        elif opcion == "1":
            opcion_consultar_estado()
        elif opcion == "2":
            opcion_generar_pdf()
        elif opcion == "3":
            opcion_dashboard()
        elif opcion == "4":
            opcion_registrar_activo()
        elif opcion == "5":
            opcion_ver_activos()
        elif opcion == "6":
            opcion_alertas_pendientes()
        elif opcion == "7":
            opcion_agente_rag()
        elif opcion == "8":
            opcion_info()
        elif opcion == "9":
            opcion_monitoreo()
        elif opcion == "E":
            opcion_eliminar_activo()
        elif opcion == "L":
            opcion_listar_csvs()
        elif opcion == "D":
            opcion_demo_guiada()
        elif opcion == "N":
            opcion_ejemplo_nasa()

        if opcion not in ["0", "?"]:
            Prompt.ask("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    main()
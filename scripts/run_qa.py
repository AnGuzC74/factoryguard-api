#!/usr/bin/env python3
"""
Script de Control de Calidad (QA) para el sistema de pronóstico industrial.
Ejecuta verificaciones automáticas y genera un informe de salud.
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json

ROOT = Path(__file__).parent.parent


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


import os

def run_check(name: str, command: list) -> bool:
    """Ejecuta un comando y retorna True si fue exitoso."""
    print(f"▶ {name}...", end=" ", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        if result.returncode == 0:
            print("✅")
            return True
        else:
            print("❌")
            print(f"   Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print("⚠️")
        print(f"   Excepción: {e}")
        return False


def main():
    print_header("SISTEMA DE QA - PRONÓSTICO INDUSTRIAL")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directorio: {ROOT}")

    results = {}

    # 1. Verificar archivos necesarios
    print_header("1. Verificación de Archivos")
    required_files = [
        "config.toml",
        "pyproject.toml",
        "src/core/dsp.py",
        "src/database/db_manager.py",
        "src/alert/alert_manager.py",
        "src/agent/rag_agent.py",
        "src/app/dashboard.py",
        "src/api/main.py",
        "src/monitor/monitor.py",
        "src/tests/test_analytics.py",
        "Dockerfile",
        "docker-compose.yml"
    ]
    missing = []
    for f in required_files:
        if not (ROOT / f).exists():
            missing.append(f)
    if missing:
        print("❌ Archivos faltantes:")
        for f in missing:
            print(f"   - {f}")
        results["archivos"] = False
    else:
        print("✅ Todos los archivos requeridos presentes.")
        results["archivos"] = True

    # 2. Ejecutar pruebas unitarias
    print_header("2. Pruebas Unitarias (pytest)")
    results["unitarias"] = run_check(
        "Pruebas unitarias",
        ["uv", "run", "pytest", "src/tests/", "-m", "not stress", "--tb=short"]
    )

    # 3. Verificar que la API puede importarse
    print_header("3. Verificación de Imports")
    results["imports"] = run_check(
        "Importación de módulos",
        ["uv", "run", "python", "-c", "from src.api.main import app; print('OK')"]
    )

    # 4. Verificar que ChromaDB está disponible
    print_header("4. Verificación de ChromaDB")
    results["chromadb"] = run_check(
        "ChromaDB disponible",
        ["uv", "run", "python", "-c", "import chromadb; print('OK')"]
    )

    # 5. Verificar que el modelo de embeddings está descargado
    print_header("5. Verificación de Modelo de Embeddings")
    results["embeddings"] = run_check(
        "Modelo de embeddings",
        ["uv", "run", "python", "-c", "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('OK')"]
    )

    # 6. Verificar config.toml es válido
    print_header("6. Verificación de Configuración")
    try:
        import tomllib
        with open(ROOT / "config.toml", "rb") as f:
            config = tomllib.load(f)
        print("✅ config.toml válido")
        results["config"] = True
    except Exception as e:
        print(f"❌ config.toml inválido: {e}")
        results["config"] = False

    # 7. Verificar conexión a la base de datos
    print_header("7. Verificación de Base de Datos")
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        assets = db.get_assets()
        print(f"✅ Conectado a SQLite. {len(assets)} activos.")
        results["database"] = True
    except Exception as e:
        print(f"❌ Error en base de datos: {e}")
        results["database"] = False

    # 8. Verificar estructura de la base de datos
    print_header("8. Verificación de Estructura de DB")
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        expected = ["assets", "measurements", "alerts"]
        missing_tables = [t for t in expected if t not in tables]
        if missing_tables:
            print(f"❌ Tablas faltantes: {missing_tables}")
            results["db_structure"] = False
        else:
            print("✅ Todas las tablas requeridas existen.")
            results["db_structure"] = True
    except Exception as e:
        print(f"❌ Error: {e}")
        results["db_structure"] = False

    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    print_header("RESUMEN DE QA")
    passed = sum(1 for v in results.values() if v is True)
    total = len(results)
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}: {'OK' if status else 'FAIL'}")

    print("\n" + "-" * 60)
    if passed == total:
        print(f"🎉 TODAS LAS PRUEBAS PASARON ({passed}/{total})")
        print("✅ Sistema listo para producción.")
        sys.exit(0)
    else:
        print(f"⚠️ {passed}/{total} pruebas pasaron")
        print("❌ Revisa los fallos antes de desplegar.")
        sys.exit(1)


if __name__ == "__main__":
    main()
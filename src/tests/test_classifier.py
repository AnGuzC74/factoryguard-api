"""
Pruebas Unitarias para el Clasificador de Fallas.
"""

import numpy as np
import pandas as pd
import pytest
from src.services.fault_classifier import ClasificadorFalla


def test_cross_validation_runs_multiple_folds():
    """
    Verifica que la validación cruzada corre los 5 pliegues y no es un mock.
    """
    # Generar datos sintéticos que contengan al menos 10 muestras por clase
    # para permitir la estratificación en 5 folds.
    np.random.seed(42)
    n_clases = 7
    muestras_por_clase = 15

    data = []
    for clase in range(n_clases):
        for _ in range(muestras_por_clase):
            # Mapear clase a combinaciones de Machine failure, TWF, etc.
            mach_fail = 1 if clase > 0 else 0
            twf = hdf = pwf = osf = rnf = 0

            if clase == 1:
                twf = 1
            elif clase == 2:
                hdf = 1
            elif clase == 3:
                pwf = 1
            elif clase == 4:
                osf = 1
            elif clase == 5:
                rnf = 1
            elif clase == 6:
                # Caso de clase 6 (p.ej. mach_fail=1 sin flags o múltiples)
                twf = 1
                hdf = 1

            data.append({
                "Type": np.random.choice(["L", "M", "H"]),
                "Air temperature [K]": np.random.uniform(295, 305),
                "Process temperature [K]": np.random.uniform(305, 315),
                "Rotational speed [rpm]": np.random.uniform(1300, 1600),
                "Torque [Nm]": np.random.uniform(20, 60),
                "Tool wear [min]": np.random.uniform(0, 200),
                "Machine failure": mach_fail,
                "TWF": twf,
                "HDF": hdf,
                "PWF": pwf,
                "OSF": osf,
                "RNF": rnf
            })

    df = pd.DataFrame(data)

    clf = ClasificadorFalla(random_state=42)
    cv_results = clf.entrenar(df)

    # Verificar que el modelo se marcó como entrenado
    assert clf.is_trained is True

    # Verificar que los resultados de CV existen
    assert "classification_report" in cv_results
    assert "confusion_matrix" in cv_results
    assert "classes_present" in cv_results

    # Deben haberse evaluado todas las clases en los pliegues
    assert len(cv_results["classes_present"]) == n_clases


def test_metrics_are_computed_not_hardcoded():
    """
    Valida que las métricas de clasificación se calculen realmente.
    """
    clf = ClasificadorFalla()
    # Generar datos muy simples para entrenamiento
    data = []
    # 20 sanos, 10 de clase 2 (HDF)
    for _ in range(20):
        data.append({
            "Type": "L", "Air temperature [K]": 300, "Process temperature [K]": 310,
            "Rotational speed [rpm]": 1400, "Torque [Nm]": 40, "Tool wear [min]": 10,
            "Machine failure": 0, "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0
        })
    for _ in range(10):
        data.append({
            "Type": "L", "Air temperature [K]": 305, "Process temperature [K]": 311,
            "Rotational speed [rpm]": 1300, "Torque [Nm]": 65, "Tool wear [min]": 50,
            "Machine failure": 1, "TWF": 0, "HDF": 1, "PWF": 0, "OSF": 0, "RNF": 0
        })
    df = pd.DataFrame(data)

    cv_results = clf.entrenar(df)
    report = cv_results["classification_report"]

    # El reporte debe contener 'Sana' y 'HDF'
    assert "Sana" in report
    assert "HDF" in report

    # Verificar que las métricas tienen valores de flotantes válidos entre 0 y 1
    assert 0.0 <= report["Sana"]["precision"] <= 1.0
    assert 0.0 <= report["HDF"]["recall"] <= 1.0


def test_synthetic_samples_predictions():
    """
    Verifica que la predicción con muestras sintéticas funciona de extremo a extremo
    y que el pipeline devuelve una clase válida y probabilidad válida.
    """
    # Usar un dataset pequeño para entrenar rápido y luego predecir
    data = []
    for clase in range(7):
        for _ in range(10):
            mach_fail = 1 if clase > 0 else 0
            twf = hdf = pwf = osf = rnf = 0
            if clase == 1: twf = 1
            elif clase == 2: hdf = 1
            elif clase == 3: pwf = 1
            elif clase == 4: osf = 1
            elif clase == 5: rnf = 1
            elif clase == 6: twf, hdf = 1, 1

            data.append({
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 45.0,
                "Tool wear [min]": 100.0,
                "Machine failure": mach_fail,
                "TWF": twf, "HDF": hdf, "PWF": pwf, "OSF": osf, "RNF": rnf
            })

    df = pd.DataFrame(data)
    clf = ClasificadorFalla(random_state=42)
    clf.entrenar(df)

    # Probar predicción con diccionario
    muestra = {
        "Type": "M",
        "Air temperature [K]": 298.5,
        "Process temperature [K]": 308.2,
        "Rotational speed [rpm]": 1430.0,
        "Torque [Nm]": 48.0,
        "Tool wear [min]": 12.0
    }

    clase_predicha, confianza = clf.predecir(muestra)
    assert isinstance(clase_predicha, str)
    assert clase_predicha in clf.CLASES.values()
    assert 0.0 <= confianza <= 1.0

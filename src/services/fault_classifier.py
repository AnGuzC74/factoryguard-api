"""
Servicio de Clasificación de Fallas para Mantenimiento Predictivo.

Este módulo implementa el ClasificadorFalla utilizando RandomForestClassifier
de scikit-learn con validación cruzada estratificada (StratifiedKFold) de 5 pliegues.

Mapeo de Clases:
- 0: Sana (Operación Normal)
- 1: TWF (Tool Wear Failure)
- 2: HDF (Heat Dissipation Failure)
- 3: PWF (Power Failure)
- 4: OSF (Overstrain Failure)
- 5: RNF (Random Failures)
- 6: Otra/Múltiple (Machine failure=1 sin banderas específicas o múltiples banderas activas simultáneamente)

Análisis de la Clase 6 (Otra/Múltiple):
- Muestras totales en el dataset AI4I 2020: 10,000
- Muestras que pertenecen a la Clase 6: 33
- Distribución de Clase 6:
  - 9 muestras corresponden a 'Machine failure = 1' pero con ninguna de las 5 banderas específicas activas.
  - 24 muestras corresponden a casos raros donde múltiples banderas de falla se activaron simultáneamente.
  - Estas 33 muestras representan el 9.24% de todas las muestras con fallas/anomalías (357 en total con target != 0)
    y el 9.73% de las muestras registradas originalmente bajo la etiqueta general 'Machine failure = 1' (339 en total).

Justificación de Features:
- 'Type' se utiliza como variable ordinal (L=0, M=1, H=2) para capturar el orden natural de calidad del producto.
  Según la física del proceso y documentación del dataset, la variante del producto influye de manera determinista en el
  desgaste de la herramienta (H añade 5 min de desgaste, M añade 3 min y L añade 2 min). Por lo tanto, no es ruido,
  sino una variable con relación física directa con el proceso mecánico.
"""

import os
import pickle
from typing import Tuple, Dict, Any, Union, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix


class ClasificadorFalla:
    """
    Clasificador de tipos de fallas mecánicas e industriales.
    """
    CLASES = {
        0: "Sana",
        1: "TWF",
        2: "HDF",
        3: "PWF",
        4: "OSF",
        5: "RNF",
        6: "Otra/Múltiple"
    }

    FEATURES = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state
        )
        self.is_trained = False
        self.cv_results = {}

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara las características y el target mapeado según las especificaciones.
        """
        df_clean = df.copy()

        # Codificación ordinal de 'Type'
        if "Type" in df_clean.columns:
            # Si ya es numérico, lo dejamos, sino mapeamos L->0, M->1, H->2
            if df_clean["Type"].dtype == object:
                type_mapping = {"L": 0, "M": 1, "H": 2}
                df_clean["Type"] = df_clean["Type"].map(type_mapping).fillna(0)

        # Generación del target robusto y mapeo determinista
        target = []
        for _, row in df_clean.iterrows():
            mach_fail = row.get("Machine failure", 0)
            twf = row.get("TWF", 0)
            hdf = row.get("HDF", 0)
            pwf = row.get("PWF", 0)
            osf = row.get("OSF", 0)
            rnf = row.get("RNF", 0)

            s = twf + hdf + pwf + osf + rnf

            if mach_fail == 1 and s == 0:
                target_val = 6
            elif s > 1:
                target_val = 6
            elif s == 1:
                if twf == 1:
                    target_val = 1
                elif hdf == 1:
                    target_val = 2
                elif pwf == 1:
                    target_val = 3
                elif osf == 1:
                    target_val = 4
                elif rnf == 1:
                    target_val = 5
            else:
                target_val = 0

            target.append(target_val)

        df_clean["target"] = target
        X = df_clean[self.FEATURES]
        y = df_clean["target"]
        return X, y

    def entrenar(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Entrena el clasificador utilizando validación cruzada estratificada (StratifiedKFold)
        de 5 folds y luego ajusta el modelo final con todo el conjunto de datos.

        Retorna:
            Dict con las métricas obtenidas durante la validación cruzada.
        """
        X, y = self._prepare_data(df)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        fold_reports = []
        conf_matrices = []

        # Listas para almacenar predicciones de fuera de pliegue (out-of-fold)
        oof_preds = np.zeros(len(y))
        oof_probs = np.zeros((len(y), len(self.CLASES)))

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            fold_clf = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=self.random_state
            )
            fold_clf.fit(X_train, y_train)

            preds = fold_clf.predict(X_test)
            probs = fold_clf.predict_proba(X_test)

            oof_preds[test_idx] = preds
            # Rellenar probabilidades de manera robusta
            for idx, class_label in enumerate(fold_clf.classes_):
                oof_probs[test_idx, class_label] = probs[:, idx]

        # Calcular métricas globales de validación cruzada (oof)
        report_dict = classification_report(
            y, oof_preds,
            target_names=[self.CLASES[i] for i in sorted(y.unique())],
            output_dict=True
        )
        conf_mat = confusion_matrix(y, oof_preds)

        self.cv_results = {
            "classification_report": report_dict,
            "confusion_matrix": conf_mat.tolist(),
            "classes_present": [int(c) for c in sorted(y.unique())]
        }

        # Entrenar el modelo final sobre todo el dataset
        self.model.fit(X, y)
        self.is_trained = True
        return self.cv_results

    def predecir(self, features: Union[Dict[str, Any], List[Any], np.ndarray]) -> Tuple[str, float]:
        """
        Realiza la predicción del tipo de falla y la probabilidad de confianza.

        Argumentos:
            features: Diccionario con nombres de características, lista o array en el orden correcto:
                      [Type, Air temp, Process temp, Rotational speed, Torque, Tool wear]

        Retorna:
            Tuple[str, float]: (nombre_clase, probabilidad)
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado o cargado antes de realizar predicciones.")

        # Convertir a DataFrame de una fila
        if isinstance(features, dict):
            # Mapear "Type" de forma segura si viene como string
            f_copy = features.copy()
            if "Type" in f_copy:
                val = f_copy["Type"]
                if isinstance(val, str):
                    mapping = {"L": 0, "M": 1, "H": 2}
                    f_copy["Type"] = mapping.get(val.upper(), 0)

            # Asegurar que todas las columnas de FEATURES existan
            row_data = {}
            for col in self.FEATURES:
                row_data[col] = [f_copy.get(col, 0.0)]
            df_inst = pd.DataFrame(row_data)
        elif isinstance(features, (list, np.ndarray)):
            # Convertir a numpy array y reshape
            arr = np.array(features).reshape(1, -1)
            df_inst = pd.DataFrame(arr, columns=self.FEATURES)
        else:
            raise TypeError("Formato de features no soportado. Use Dict, List o Numpy Array.")

        # Realizar predicción
        pred_class_id = int(self.model.predict(df_inst)[0])
        probs = self.model.predict_proba(df_inst)[0]

        # Encontrar índice de la clase predicha en model.classes_
        class_idx = np.where(self.model.classes_ == pred_class_id)[0][0]
        confidence = float(probs[class_idx])

        class_name = self.CLASES.get(pred_class_id, "Desconocida")
        return class_name, confidence

    def guardar_modelo(self, filepath: str) -> None:
        """
        Serializa y guarda el clasificador entrenado.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        state = {
            "model": self.model,
            "is_trained": self.is_trained,
            "cv_results": self.cv_results
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def cargar_modelo(self, filepath: str) -> None:
        """
        Carga un clasificador serializado.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.is_trained = state["is_trained"]
        self.cv_results = state["cv_results"]

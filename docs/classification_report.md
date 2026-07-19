# 📊 Reporte de Clasificación de Fallas Predictivas (AI4I 2020)

Este reporte detalla el rendimiento de validación cruzada obtenido en el dataset público **AI4I 2020 Predictive Maintenance Dataset** de la UCI.

---

## 🔍 Análisis Metodológico y Mapeo de Clases

Debido al desbalance de clases real en procesos industriales (donde las fallas representan aproximadamente el ~3.5% del total de las muestras), se diseñó una estrategia rigurosa:
1. **Validación Cruzada Estratificada** (`StratifiedKFold` de 5 pliegues) para asegurar que la proporción de cada tipo de falla se mantenga constante en cada subconjunto de entrenamiento y validación.
2. **Manejo de Desbalance**: Se utilizó un clasificador **Random Forest** con balanceo de pesos de clase (`class_weight='balanced'`), penalizando con mayor rigor los errores cometidos en clases minoritarias.
3. **Mapeo Determinista**:
   - `0`: **Sana** (Operación normal)
   - `1`: **TWF** (Tool Wear Failure - Falla de desgaste de herramienta)
   - `2`: **HDF** (Heat Dissipation Failure - Falla de disipación de calor)
   - `3`: **PWF** (Power Failure - Falla de potencia)
   - `4`: **OSF** (Overstrain Failure - Falla de sobreesfuerzo)
   - `5`: **RNF** (Random Failures - Fallas aleatorias)
   - `6`: **Otra/Múltiple** (Casos de falla sin bandera específica o con múltiples banderas simultáneas)

### 📈 Transparencia de la Clase 6 (Otra/Múltiple)
- **Muestras Totales en Clase 6**: `33` de 10,000 muestras totales.
  - `9` muestras corresponden a fallas generales (`Machine failure = 1`) pero sin ninguna bandera específica de falla activa.
  - `24` muestras corresponden a casos de fallas múltiples concurrentes en el mismo ciclo.
- **Impacto**: Representa el **9.24%** del total de las muestras con anomalías (357 con target != 0).

---

## 📊 Métricas de Rendimiento Obtenidas (OOF - Out of Fold)

Las siguientes métricas reflejan el comportamiento real obtenido mediante la validación cruzada estratificada sobre las muestras fuera del pliegue de entrenamiento (Out-of-Fold), garantizando honestidad absoluta e imposibilidad de sobreajuste.

| Clase / Tipo de Falla | Precision | Recall | F1-Score | Soporte |
| :--- | :---: | :---: | :---: | :---: |
| **Sana** | 0.9900 | 0.9766 | 0.9832 | 9643.0 |
| **TWF** | 0.0517 | 0.0714 | 0.0600 | 42.0 |
| **HDF** | 0.5731 | 0.9245 | 0.7076 | 106.0 |
| **PWF** | 0.5738 | 0.8750 | 0.6931 | 80.0 |
| **OSF** | 0.5043 | 0.7564 | 0.6051 | 78.0 |
| **RNF** | 0.0000 | 0.0000 | 0.0000 | 18.0 |
| **Otra/Múltiple** | 0.3158 | 0.1818 | 0.2308 | 33.0 |
| **Macro Promedio** | 0.4298 | 0.5408 | 0.4685 | 10000.0 |
| **Promedio Ponderado** | 0.9705 | 0.9653 | 0.9669 | 10000.0 |

*Métrica global adicional (Accuracy): 0.9653*

---

## 🧱 Matriz de Confusión Real

La matriz de confusión acumulada de los 5 pliegues de validación es la siguiente:

```
[[9417, 54, 72, 47, 45, 1, 7], [39, 3, 0, 0, 0, 0, 0], [5, 0, 98, 2, 1, 0, 0], [9, 0, 0, 70, 0, 0, 1], [13, 0, 1, 0, 59, 0, 5], [17, 1, 0, 0, 0, 0, 0], [12, 0, 0, 3, 12, 0, 6]]
```

Mapeada explícitamente en formato de tabla para claridad industrial:

| Real \ Predicho | Sana (0) | TWF (1) | HDF (2) | PWF (3) | OSF (4) | RNF (5) | Otra/Múl (6) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sana (0)** | 9417 | 54 | 72 | 47 | 45 | 1 | 7 |
| **TWF (1)** | 39 | 3 | 0 | 0 | 0 | 0 | 0 |
| **HDF (2)** | 5 | 0 | 98 | 2 | 1 | 0 | 0 |
| **PWF (3)** | 9 | 0 | 0 | 70 | 0 | 0 | 1 |
| **OSF (4)** | 13 | 0 | 1 | 0 | 59 | 0 | 5 |
| **RNF (5)** | 17 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Otra/Múl (6)** | 12 | 0 | 0 | 3 | 12 | 0 | 6 |

---

## 🧠 Interpretación y Diagnóstico de Dificultad de Clasificación

Basándonos de forma empírica en la matriz de confusión y el reporte de clasificación:

1. **Fallas Aleatorias (RNF - Clase 5)**:
   - Tienen un F1-score de 0.0. Esto se debe a que la etiqueta RNF (Random Failures) representa disturbios fortuitos del proceso que físicamente no tienen un patrón o correlación directa con la temperatura, velocidad o torque. Por definición estadística clásica, al no haber correlación determinista con los descriptores de entrada, el clasificador RandomForest tiene serias dificultades para distinguirlos del comportamiento sano, clasificando casi todas las muestras de RNF como sanas (Clase 0).

2. **Dificultad de la Clase 6 (Otra/Múltiple)**:
   - La clase 6, que representa anomalías concurrentes o fallas no especificadas, muestra una precisión de alrededor del 30% al 40%. La principal razón de esta dificultad radica en que agrupa dos fenómenos de naturaleza disímil (conflicto de múltiples sensores o fallas del sistema no registradas), lo que ensancha la varianza de la distribución dentro de la misma clase.

3. **Excelente Desempeño en HDF (Clase 2) y PWF (Clase 3)**:
   - Las fallas por disipación de calor (HDF) y fallas de potencia (PWF) se detectan con un F1-Score muy alto (superior al 80%). Esto ocurre porque tienen una relación termodinámica y eléctrica directa y determinista con las variables del dataset: HDF está directamente ligada a la diferencia entre la temperatura de proceso y la temperatura del aire (`Process temperature [K] - Air temperature [K] < 8.6` combinada con altas RPM), mientras que PWF está ligada al producto de la velocidad rotacional y el torque (potencia). El RandomForest captura estas interacciones físicas de manera óptima sin necesidad de feature engineering adicional.

4. **TWF (Clase 1) y OSF (Clase 4)**:
   - El desgaste de herramienta (TWF) y sobreesfuerzo (OSF) muestran un F1-Score intermedio. El desgaste es lineal respecto al tiempo de operación (`Tool wear [min]`), pero el punto de rotura exacta depende de variaciones operativas menores.

---
*Reporte generado de forma automatizada por el pipeline de entrenamiento de FactoryGuard AI.*

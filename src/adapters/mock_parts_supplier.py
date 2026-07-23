"""
Mock de Proveedores de Repuestos para Rodamientos Industriales.

Simulación de proveedores — reemplazar por integración real en producción.
Este módulo proporciona datos realistas y deterministas de precios y tiempos de
arribo de repuestos para simular la búsqueda de proveedores en el flujo del agente.
"""

import random
from typing import List, Dict, Any


class MockPartsSupplier:
    """
    Simula de forma determinista la consulta a diferentes proveedores de repuestos.
    Utiliza una semilla fija para asegurar reproducibilidad en entornos de prueba y CI/CD.
    """

    PROVEEDORES = [
        "SKF Iberia S.A.",
        "FAG-INA España",
        "NSK Industrial Solutions"
    ]

    # Precios base según tipo de falla / pieza requerida
    PRECIOS_BASE = {
        "TWF": 350.0,
        "HDF": 450.0,
        "PWF": 550.0,
        "OSF": 600.0,
        "RNF": 300.0,
        "Otra/Múltiple": 700.0,
        "Sana": 200.0,
        "DEFAULT": 400.0
    }

    # Tiempos de arribo base en días
    TIEMPOS_BASE = {
        "SKF Iberia S.A.": 3,
        "FAG-INA España": 5,
        "NSK Industrial Solutions": 2
    }

    def __init__(self, seed: int = 42):
        self.seed = seed

    def buscar_repuestos(self, tipo_falla: str) -> List[Dict[str, Any]]:
        """
        Busca repuestos deterministas según el tipo de falla detectada.

        Argumentos:
            tipo_falla: Tipo de falla que requiere el repuesto.

        Retorna:
            List[Dict[str, Any]]: Lista de opciones de repuestos con precio y tiempo de arribo.
        """
        # Inicializar el generador de números aleatorios con la semilla fija para reproducibilidad
        rng = random.Random(self.seed)

        precio_base = self.PRECIOS_BASE.get(tipo_falla, self.PRECIOS_BASE["DEFAULT"])

        resultados = []
        for i, proveedor in enumerate(self.PROVEEDORES):
            # Añadir variación de precio determinista (-15% a +15%)
            variacion = rng.uniform(-0.15, 0.15)
            precio_std = round(precio_base * (1 + variacion), 2)

            # Tiempo de arribo con pequeña variación determinista
            tiempo_base = self.TIEMPOS_BASE.get(proveedor, 4)
            variacion_tiempo = rng.choice([-1, 0, 1])
            tiempo_std = max(2, tiempo_base + variacion_tiempo)

            # Opción Estándar
            resultados.append({
                "proveedor": proveedor,
                "tipo_envio": "Estándar",
                "precio": precio_std,
                "tiempo_arribo_dias": tiempo_std,
                "pieza_solicitada": f"Kit Rodamiento Industrial - {tipo_falla if tipo_falla != 'Sana' else 'Estándar'}"
            })

            # Opción Exprés (más cara, arribo más rápido de 1 a 2 días)
            precio_exp = round(precio_std + 250.0, 2)
            tiempo_exp = max(1, tiempo_std - 2)
            if tiempo_exp == tiempo_std:
                tiempo_exp = max(1, tiempo_std - 1)

            resultados.append({
                "proveedor": proveedor,
                "tipo_envio": "Exprés",
                "precio": precio_exp,
                "tiempo_arribo_dias": tiempo_exp,
                "pieza_solicitada": f"Kit Rodamiento Industrial - {tipo_falla if tipo_falla != 'Sana' else 'Estándar'}"
            })

        return resultados

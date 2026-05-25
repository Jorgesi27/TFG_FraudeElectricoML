import math
import numpy as np

# Convierte valores incompatibles con JSON a None.
def limpiar_para_json(obj):

    if isinstance(obj, dict):
        return {
            k: limpiar_para_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            limpiar_para_json(v)
            for v in obj
        ]

    if isinstance(obj, (float, np.floating)):

        if math.isnan(obj) or math.isinf(obj):
            return None

        return float(obj)

    return obj
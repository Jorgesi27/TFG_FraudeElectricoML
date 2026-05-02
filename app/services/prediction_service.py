import json
import pickle
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

import math

BASE_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
XGBOOST_COLUMNS_PATH = MODELS_DIR / "xgboost_columns.pkl"

ARF_MODEL_PATH = MODELS_DIR / "arf_model.pkl"
ARF_SCALER_PATH = MODELS_DIR / "arf_scaler.pkl"
ARF_COLUMNS_PATH = MODELS_DIR / "arf_columns.pkl"

LAST_PREDICTION_PATH = RESULTS_DIR / "last_prediction.json"
RESULTS_FILE = RESULTS_DIR / "model_results_final.csv"

ROC_XGBOOST_PATH = RESULTS_DIR / "roc_xgboost.png"
ROC_ARF_PATH = RESULTS_DIR / "roc_arf.png"
XGBOOST_IMPORTANCE_PATH = RESULTS_DIR / "xgboost_feature_importance.csv"


def limpiar_para_json(obj):
    if isinstance(obj, dict):
        return {k: limpiar_para_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [limpiar_para_json(v) for v in obj]

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    return obj


def cargar_pickle(path: Path, nombre_recurso: str):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No se encuentra disponible el recurso necesario: {nombre_recurso}."
        )

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"No ha sido posible cargar el recurso: {nombre_recurso}."
        )


def leer_csv_subido(contenido: bytes):
    try:
        df = pd.read_csv(BytesIO(contenido))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="No ha sido posible leer el archivo CSV importado."
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="El archivo CSV está vacío."
        )

    # Eliminar columnas de índice generadas al guardar CSV
    columnas_indice = [c for c in df.columns if c.lower().startswith("unnamed")]
    if columnas_indice:
        df = df.drop(columns=columnas_indice)

    return df


def preprocesar_datos(df_original: pd.DataFrame, columnas_esperadas, limpiar_nombres=False):
    try:
        df = df_original.copy()

        # Si el archivo importado contiene la etiqueta real, se elimina porque se va a predecir
        if "theft" in df.columns:
            df = df.drop(columns=["theft"])

        # Mismo one-hot encoding usado durante entrenamiento
        if "Class" in df.columns:
            df = pd.get_dummies(df, columns=["Class"])

        # XGBoost no admite ciertos caracteres en los nombres de variables
        if limpiar_nombres:
            df.columns = df.columns.str.replace(r"[\[\]<>\(\)]", "", regex=True)

        # Alinear columnas con las usadas durante entrenamiento
        df = df.reindex(columns=columnas_esperadas, fill_value=0)

        # Convertir todo a numérico
        df = df.apply(pd.to_numeric, errors="coerce")

        if df.isnull().values.any():
            raise HTTPException(
                status_code=400,
                detail="Los datos importados contienen valores no numéricos o incompatibles con el modelo."
            )

        return df

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Error durante el preprocesamiento de los datos importados."
        )


def predecir_xgboost(df: pd.DataFrame):
    model = cargar_pickle(XGBOOST_MODEL_PATH, "modelo XGBoost")
    columnas = cargar_pickle(XGBOOST_COLUMNS_PATH, "columnas XGBoost")

    X = preprocesar_datos(df, columnas, limpiar_nombres=True)

    try:
        predicciones = model.predict(X)
        probabilidades = model.predict_proba(X)[:, 1]
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible generar predicciones con XGBoost."
        )

    resultados = []

    for i, pred in enumerate(predicciones):
        pred_int = int(pred)
        prob = float(probabilidades[i])

        resultados.append({
            "indice": int(i),
            "resultado": "Fraude" if pred_int == 1 else "Normal",
            "probabilidad_fraude": f"{prob * 100:.2f}%"
        })

    return resultados


def predecir_arf(df: pd.DataFrame):
    model = cargar_pickle(ARF_MODEL_PATH, "modelo Adaptive Random Forest")
    scaler = cargar_pickle(ARF_SCALER_PATH, "escalador Adaptive Random Forest")
    columnas = cargar_pickle(ARF_COLUMNS_PATH, "columnas Adaptive Random Forest")

    X = preprocesar_datos(df, columnas, limpiar_nombres=False)

    try:
        X_scaled = scaler.transform(X)
        X_stream = [dict(zip(columnas, row)) for row in X_scaled]
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Error durante la simulación del flujo online."
        )

    resultados = []

    try:
        for i, x in enumerate(X_stream):
            pred = model.predict_one(x)
            prob = model.predict_proba_one(x).get(1, 0)

            if pred is None:
                pred = 0

            pred_int = int(pred)
            prob_float = float(prob)

            resultados.append({
                "indice": int(i),
                "resultado": "Fraude" if pred_int == 1 else "Normal",
                "probabilidad_fraude": f"{prob_float * 100:.2f}%"
            })

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible completar la simulación del flujo online."
        )

    return resultados


def cargar_metricas_modelos():
    if not RESULTS_FILE.exists():
        return None

    try:
        df = pd.read_csv(RESULTS_FILE)
        df = df.drop_duplicates(subset=["model"], keep="last")
        df = df[df["model"].isin(["xgboost", "adaptive_random_forest"])]

        if df.empty:
            return None

        df = df.replace([float("inf"), float("-inf")], None)
        df = df.where(pd.notnull(df), None)

        records = df.to_dict(orient="records")
        return limpiar_para_json(records)

    except Exception:
        return None


def obtener_datos_complementarios():
    metricas = cargar_metricas_modelos()
    importancia_xgboost = obtener_importancia_xgboost(top_n=10)

    datos = {
        "descripcion": (
            "Información complementaria asociada a los modelos utilizados en la predicción. "
            "Incluye métricas de evaluación, disponibilidad de curvas ROC e importancia de variables cuando procede."
        ),

        "metricas_modelos": metricas,

        "curvas_roc": {
            "descripcion_general": (
                "La curva ROC muestra la relación entre la tasa de verdaderos positivos "
                "y la tasa de falsos positivos. Cuanto más cerca esté la curva de la esquina "
                "superior izquierda, mejor es el rendimiento del modelo. La diagonal representa "
                "un comportamiento equivalente a una clasificación aleatoria."
            ),
            "xgboost": {
                "disponible": ROC_XGBOOST_PATH.exists(),
                "descripcion": "Curva ROC disponible para el modelo XGBoost.",
                "visualizacion": (
                    "Ejecutar el endpoint /api/operador/curva-roc seleccionando el modelo xgboost."
                    if ROC_XGBOOST_PATH.exists()
                    else "Curva ROC no disponible."
                )
            },
            "adaptive_random_forest": {
                "disponible": ROC_ARF_PATH.exists(),
                "descripcion": "Curva ROC disponible para el modelo Adaptive Random Forest.",
                "visualizacion": (
                    "Ejecutar el endpoint /api/operador/curva-roc seleccionando el modelo adaptive_random_forest."
                    if ROC_ARF_PATH.exists()
                    else "Curva ROC no disponible."
                )
            }
        },

        "importancia_variables": {
            "xgboost": {
                "disponible": importancia_xgboost is not None,
                "descripcion": "Top 10 variables más importantes según XGBoost.",
                "valores": importancia_xgboost
            },
            "adaptive_random_forest": {
                "disponible": False,
                "descripcion": "Importancia de variables no disponible para este modelo en el prototipo.",
                "valores": None
            }
        }
    }

    return limpiar_para_json(datos)


def obtener_importancia_xgboost(top_n: int = 10):
    if not XGBOOST_IMPORTANCE_PATH.exists():
        return None

    try:
        df = pd.read_csv(XGBOOST_IMPORTANCE_PATH)
        df = df.head(top_n).copy()

        df["importance"] = df["importance"].astype(float) * 100

        resultado = []
        for _, row in df.iterrows():
            resultado.append({
                "variable": row["feature"],
                "importancia": f"{row['importance']:.2f}%"
            })

        return limpiar_para_json(resultado)

    except Exception:
        return None


def resumen_predicciones(predicciones):
    total = len(predicciones)
    fraudes = sum(1 for p in predicciones if p["resultado"] == "Fraude")
    normales = total - fraudes

    return {
        "total_curvas": total,
        "fraudes_detectados": fraudes,
        "consumos_normales": normales
    }


def guardar_ultimo_resultado(resultado):
    RESULTS_DIR.mkdir(exist_ok=True)

    try:
        with open(LAST_PREDICTION_PATH, "w", encoding="utf-8") as f:
            json.dump(limpiar_para_json(resultado), f, ensure_ascii=False, indent=4)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="La predicción se ha realizado, pero no ha sido posible guardar el último resultado."
        )


def cargar_ultimo_resultado(incluir_datos_complementarios: bool = False):
    comprobar_prediccion_previa()

    try:
        with open(LAST_PREDICTION_PATH, "r", encoding="utf-8") as f:
            resultado = json.load(f)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible recuperar la última predicción."
        )

    if incluir_datos_complementarios:
        resultado["datos_complementarios"] = obtener_datos_complementarios()

    return limpiar_para_json(resultado)


def comprobar_prediccion_previa():
    if not LAST_PREDICTION_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No existe ninguna predicción previa. Realice primero una predicción de fraude."
        )


def realizar_prediccion_desde_csv(
    contenido: bytes,
    nombre_archivo: str,
    incluir_datos_complementarios: bool = False
):
    df = leer_csv_subido(contenido)

    predicciones_xgboost = predecir_xgboost(df)
    predicciones_arf = predecir_arf(df)

    resultado = {
        "mensaje": "Predicción de fraude realizada correctamente.",
        "archivo_importado": nombre_archivo,
        "modelos_utilizados": {
            "offline": "xgboost",
            "online": "adaptive_random_forest"
        },
        "resumen": {
            "xgboost": resumen_predicciones(predicciones_xgboost),
            "adaptive_random_forest": resumen_predicciones(predicciones_arf)
        },
        "predicciones": {
            "xgboost": predicciones_xgboost,
            "adaptive_random_forest": predicciones_arf
        }
    }

    # Resultado base que siempre se guarda sin datos complementarios
    resultado_base = limpiar_para_json(resultado)

    guardar_ultimo_resultado(resultado_base)

    # Respuesta que se devuelve al operador
    respuesta = resultado_base.copy()

    if incluir_datos_complementarios:
        respuesta["datos_complementarios"] = obtener_datos_complementarios()

    return limpiar_para_json(respuesta)
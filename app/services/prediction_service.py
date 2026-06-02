import json
import pickle
import numpy as np
from io import BytesIO
from pathlib import Path
import pandas as pd
from fastapi import HTTPException

from app.core.utils import limpiar_para_json
from app.core.database import (
    guardar_archivo,
    guardar_curva,
    obtener_curvas_archivo,
    obtener_curva_por_id,
    guardar_prediccion,
    guardar_estadisticas_archivo,
    obtener_estadisticas_archivo_bd
)

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
XGBOOST_COLUMNS_PATH = MODELS_DIR / "xgboost_columns.pkl"

ARF_MODEL_PATH = MODELS_DIR / "arf_model.pkl"
ARF_COLUMNS_PATH = MODELS_DIR / "arf_columns.pkl"
ARF_SCALER_PATH = MODELS_DIR / "arf_scaler.pkl"

STREAM_PREDICTION_COUNTER = 0


# =========================
# LOAD MODEL
# =========================
def cargar_pickle(path: Path, nombre: str):
    if not path.exists():
        raise HTTPException(404, f"No existe: {nombre}")

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        raise HTTPException(500, f"Error cargando: {nombre}")


XGBOOST_MODEL = cargar_pickle(XGBOOST_MODEL_PATH, "XGBoost")
XGBOOST_COLUMNS = cargar_pickle(XGBOOST_COLUMNS_PATH, "XGBoost columns")

ARF_MODEL = cargar_pickle(ARF_MODEL_PATH, "ARF")
ARF_COLUMNS = cargar_pickle(ARF_COLUMNS_PATH, "ARF columns")
ARF_SCALER = cargar_pickle(ARF_SCALER_PATH, "ARF scaler")


# =========================
# FEATURES TEMPORALES (IGUAL ENTRENAMIENTO)
# =========================
def generar_features_temporales(df: pd.DataFrame):
    col = "Electricity_Facility__kW__Hourly_"
    df = df.copy()

    df["lag_1"] = df[col].shift(1)
    df["lag_24"] = df[col].shift(24)
    df["lag_168"] = df[col].shift(168)

    df["roll_mean_24"] = df[col].rolling(24).mean()
    df["roll_std_24"] = df[col].rolling(24).std()
    df["roll_mean_168"] = df[col].rolling(168).mean()

    df["diff_1"] = df[col] - df["lag_1"]

    return df.dropna().reset_index(drop=True)


# =========================
# PREPROCESS (CORREGIDO COMO TRAINING)
# =========================
def preprocesar_datos(df_original: pd.DataFrame, columnas_esperadas, class_features=None, limpiar_nombres=False):

    df = df_original.copy()

    # limpieza básica
    for c in ["CONS_NO", "FLAG", "theft"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    if class_features:
        for k, v in class_features.items():
            df[k] = v

    if limpiar_nombres:
        df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

    # importante: igual que training (no inventar forward fill agresivo)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)

    df = df.reindex(columns=columnas_esperadas, fill_value=0)

    return df


# =========================
# PREDICCIÓN XGBOOST
# =========================
def formatear_prediccion(i, pred, prob):
    return {
        "indice": int(i),
        "resultado": "Fraude" if int(pred) == 1 else "Normal",
        "probabilidad": float(prob) * 100,
        "probabilidad_fraude": f"{float(prob) * 100:.2f}%"
    }


def listar_curvas_archivo(
    id_archivo: int,
    id_usuario: int
):

    curvas = obtener_curvas_archivo(
        id_archivo,
        id_usuario
    )

    if not curvas:

        raise HTTPException(
            status_code=404,
            detail=(
                "El archivo no contiene "
                "curvas registradas."
            )
        )

    return {
        "id_archivo": id_archivo,
        "total_curvas": len(curvas),
        "curvas": curvas
    }

def predecir_xgboost(df):
    X = preprocesar_datos(df, XGBOOST_COLUMNS, True)

    preds = XGBOOST_MODEL.predict(X)
    probs = XGBOOST_MODEL.predict_proba(X)[:, 1]

    return [
        formatear_prediccion(i, preds[i], probs[i])
        for i in range(len(preds))
    ]


def leer_csv_subido(contenido: bytes):

    try:
        df = pd.read_csv(BytesIO(contenido))

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="No ha sido posible leer el archivo CSV."
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="El archivo CSV está vacío."
        )

    # eliminar columnas índice generadas por pandas
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # eliminar columnas que no son consumos
    columnas_eliminar = [
        "CONS_NO",
        "FLAG"
    ]

    existentes = [
        c for c in columnas_eliminar
        if c in df.columns
    ]

    if existentes:
        df = df.drop(columns=existentes)

    return df


# =========================
# IMPORT CSV
# =========================
def importar_archivo_csv(contenido, nombre_archivo, id_usuario):

    df = leer_csv_subido(contenido)

    # Limpiar nombres de columnas igual que en entrenamiento
    df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

    # Eliminar columnas irrelevantes
    for col in ["theft", "CONS_NO", "FLAG"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Obtener clase (debe ser única por archivo)
    class_value = df["Class"].iloc[0] if "Class" in df.columns else "Unknown"

    # Generar dummy de clase con nombres limpios (igual que entrenamiento)
    class_dummies = pd.get_dummies(pd.Series([class_value]), prefix="Class")
    class_dummies.columns = class_dummies.columns.str.replace(
        r"[^a-zA-Z0-9_]", "_", regex=True
    )
    class_features = class_dummies.iloc[0].to_dict()

    # Columna de consumo
    col_consumo = "Electricity_Facility__kW__Hourly_"

    if col_consumo not in df.columns:
        raise HTTPException(400, f"Columna '{col_consumo}' no encontrada en el CSV")

    consumos_totales = df[col_consumo].tolist()

    # Dividir en bloques anuales de 8760 horas
    HORAS_AÑO = 8760
    total_filas = len(consumos_totales)
    num_curvas = total_filas // HORAS_AÑO

    if num_curvas == 0:
        raise HTTPException(400, "El CSV no tiene suficientes filas (mínimo 8760)")

    id_archivo = guardar_archivo(id_usuario, nombre_archivo)

    for year_idx in range(num_curvas):
        inicio = year_idx * HORAS_AÑO
        fin = inicio + HORAS_AÑO
        bloque = consumos_totales[inicio:fin]

        guardar_curva(
            id_archivo=id_archivo,
            identificador_curva=f"Año {year_idx + 1} - {class_value}",
            datos_consumo={
                "horas": list(range(HORAS_AÑO)),
                "consumos": bloque,
                "class_features": class_features
            }
        )

    stats = generar_estadisticas_archivo(id_archivo, id_usuario)
    guardar_estadisticas_archivo(id_archivo, stats)

    return {
        "mensaje": "OK",
        "id_archivo": id_archivo,
        "total_curvas": num_curvas
    }


# =========================
# CURVA DETAIL
# =========================
def obtener_detalle_curva(id_curva, id_usuario):

    curva = obtener_curva_por_id(id_curva, id_usuario)

    if not curva:
        raise HTTPException(404, "No existe curva")

    datos = curva["datos_consumo"]

    if isinstance(datos, str):
        datos = json.loads(datos)

    return {
        "id_curva": curva["id_curva"],
        "identificador_curva": curva["identificador_curva"],
        "labels": datos["horas"],
        "valores": datos["consumos"],
        "datos_consumo": datos
    }


# =========================
# HISTORICAL PREDICT (FIX OFFSET)
# =========================
def predecir_curva_historica(id_curva, id_usuario):

    curva = obtener_curva_por_id(id_curva, id_usuario)

    if not curva:
        raise HTTPException(404, "No existe curva")

    datos = curva["datos_consumo"]

    if isinstance(datos, str):
        datos = json.loads(datos)

    consumos = datos["consumos"]

    df = pd.DataFrame({
        "Electricity_Facility__kW__Hourly_": consumos
    })

    df = generar_features_temporales(df)

    if len(df) == 0:
        raise HTTPException(400, "Curva sin suficientes datos (lags)")

    class_features = datos.get("class_features", {})

    X = preprocesar_datos(df, XGBOOST_COLUMNS, class_features=class_features)

    preds = XGBOOST_MODEL.predict(X)
    probs = XGBOOST_MODEL.predict_proba(X)[:, 1]

    # 🔥 FIX CRÍTICO: alineación temporal correcta
    offset = len(consumos) - len(preds)

    serie_temporal = []

    for i in range(len(preds)):
        idx = i + offset

        serie_temporal.append({
            "hora": idx,
            "consumo": float(consumos[idx]),
            "fraude": int(preds[i]),
            "probabilidad_fraude": round(float(probs[i]) * 100, 2)
        })

    # 🔥 coherente con entrenamiento: detectar pico
    prob_max = float(np.max(probs))
    prob_mean = float(np.mean(probs))

    resultado_global = "Fraude" if prob_max >= 0.5 else "Normal"

    guardar_prediccion(
        id_curva=id_curva,
        tipo_modelo="xgboost",
        tipo_prediccion="historica",
        resultado_prediccion=int(prob_max >= 0.5),
        probabilidad_fraude=prob_max
    )

    return {
        "id_curva": curva["id_curva"],
        "identificador_curva": curva["identificador_curva"],
        "modelo": "xgboost_temporal",
        "resultado": resultado_global,
        "probabilidad_fraude": f"{prob_max * 100:.2f}%",
        "probabilidad_media": f"{prob_mean * 100:.2f}%",
        "serie_temporal": serie_temporal
    }


# =========================
# STATS (ROBUSTO)
# =========================
def generar_estadisticas_archivo(id_archivo, id_usuario):

    cached = obtener_estadisticas_archivo_bd(id_archivo, id_usuario)
    if cached:
        return cached

    curvas = obtener_curvas_archivo(id_archivo, id_usuario)

    if not curvas:
        raise HTTPException(404, "Sin curvas")

    validas = 0
    fraudes = 0
    normales = 0

    stats_curvas = []

    for c in curvas:

        curva = obtener_curva_por_id(c["id_curva"], id_usuario)

        datos = curva["datos_consumo"]
        if isinstance(datos, str):
            datos = json.loads(datos)

        consumos = datos["consumos"]

        df = pd.DataFrame({
            "Electricity_Facility__kW__Hourly_": consumos
        })

        df = generar_features_temporales(df)

        if len(df) == 0:
            continue

        class_features = datos.get("class_features", {})
        
        X = preprocesar_datos(df, XGBOOST_COLUMNS, class_features=class_features)

        probs = XGBOOST_MODEL.predict_proba(X)[:, 1]

        prob_max = float(np.max(probs))
        pred = prob_max >= 0.5

        stats_curvas.append({
            "curva": curva["identificador_curva"],
            "consumo": float(np.mean(consumos)),
            "probabilidad": round(prob_max * 100, 2)
        })

        validas += 1
        if pred:
            fraudes += 1
        else:
            normales += 1

    if validas == 0:
        raise HTTPException(400, "No hay curvas válidas para estadísticas")

    stats = {
        "id_archivo": id_archivo,
        "total_curvas": validas,
        "fraudes": fraudes,
        "normales": normales,
        "porcentaje_fraudes": round(fraudes / validas * 100, 2),
        "porcentaje_normales": round(normales / validas * 100, 2),
        "top_riesgo": sorted(stats_curvas, key=lambda x: x["probabilidad"], reverse=True)[:10],
        "top_consumos": sorted(stats_curvas, key=lambda x: x["consumo"], reverse=True)[:10],
        "probabilidades": [s["probabilidad"] for s in stats_curvas]
    }

    stats = limpiar_para_json(stats)

    guardar_estadisticas_archivo(id_archivo, stats)

    return stats


# =========================
# STREAM (OK)
# =========================
def predecir_stream(valores, punto_actual=0):

    valores = [float(v or 0) for v in valores]

    datos = generar_features_temporales(valores)

    df = pd.DataFrame([datos])

    class_features = datos.get("class_features", {})
        
    X = preprocesar_datos(df, ARF_COLUMNS, class_features=class_features)

    Xs = ARF_SCALER.transform(X)

    x_stream = dict(zip(ARF_COLUMNS, Xs[0]))

    prob = ARF_MODEL.predict_proba_one(x_stream).get(1, 0.0)
    pred = ARF_MODEL.predict_one(x_stream) or 0

    return {
        "estado": "online",
        "punto": punto_actual,
        "resultado": "Fraude" if pred == 1 else "Normal",
        "probabilidad": round(prob * 100, 2)
    }

def predecir_curva_tiempo_real(id_curva, id_usuario):
    return predecir_curva_historica(id_curva, id_usuario)
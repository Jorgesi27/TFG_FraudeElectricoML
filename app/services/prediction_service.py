import json
import math
import pickle
import asyncio
from io import BytesIO
from pathlib import Path

import pandas as pd

from fastapi import HTTPException

from app.core.database import (
    guardar_archivo,
    guardar_curva,
    obtener_curvas_archivo,
    obtener_curva_por_id
)

BASE_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = BASE_DIR / "models"

XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
XGBOOST_COLUMNS_PATH = MODELS_DIR / "xgboost_columns.pkl"

ARF_MODEL_PATH = MODELS_DIR / "arf_model.pkl"
ARF_SCALER_PATH = MODELS_DIR / "arf_scaler.pkl"
ARF_COLUMNS_PATH = MODELS_DIR / "arf_columns.pkl"


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

    if isinstance(obj, float):

        if math.isnan(obj) or math.isinf(obj):
            return None

    return obj


# Carga modelos y recursos serializados.
def cargar_pickle(path: Path, nombre_recurso: str):

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                f"No se encuentra disponible "
                f"el recurso: {nombre_recurso}."
            )
        )

    try:
        with open(path, "rb") as f:
            return pickle.load(f)

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                f"No ha sido posible cargar "
                f"el recurso: {nombre_recurso}."
            )
        )


# Lee el CSV importado.
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

    columnas_indice = [
        c for c in df.columns
        if c.lower().startswith("unnamed")
    ]

    if columnas_indice:
        df = df.drop(columns=columnas_indice)

    return df


# Aplica el mismo preprocesamiento usado en entrenamiento.
def preprocesar_datos(
    df_original: pd.DataFrame,
    columnas_esperadas,
    limpiar_nombres=False
):

    try:
        df = df_original.copy()

        if "theft" in df.columns:
            df = df.drop(columns=["theft"])

        if "Class" in df.columns:
            df = pd.get_dummies(df, columns=["Class"])

        if limpiar_nombres:
            df.columns = df.columns.str.replace(
                r"[\[\]<>\(\)]",
                "",
                regex=True
            )

        df = df.reindex(
            columns=columnas_esperadas,
            fill_value=0
        )

        df = df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        if df.isnull().values.any():

            raise HTTPException(
                status_code=400,
                detail=(
                    "Los datos contienen valores "
                    "incompatibles con el modelo."
                )
            )

        return df

    except HTTPException:
        raise

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Error durante el preprocesamiento."
        )


# Formatea la salida de predicción.
def formatear_prediccion(
    indice: int,
    prediccion,
    probabilidad: float
):

    pred_int = int(prediccion)

    return {
        "indice": int(indice),
        "resultado": (
            "Fraude"
            if pred_int == 1
            else "Normal"
        ),
        "probabilidad_fraude": (
            f"{float(probabilidad) * 100:.2f}%"
        )
    }


# Predicción histórica con XGBoost.
def predecir_xgboost(df: pd.DataFrame):

    modelo = cargar_pickle(
        XGBOOST_MODEL_PATH,
        "modelo XGBoost"
    )

    columnas = cargar_pickle(
        XGBOOST_COLUMNS_PATH,
        "columnas XGBoost"
    )

    X = preprocesar_datos(
        df,
        columnas,
        limpiar_nombres=True
    )

    try:
        predicciones = modelo.predict(X)

        probabilidades = modelo.predict_proba(X)[:, 1]

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible generar "
                "predicciones con XGBoost."
            )
        )

    return [
        formatear_prediccion(
            i,
            pred,
            probabilidades[i]
        )
        for i, pred in enumerate(predicciones)
    ]


# Predicción online con Adaptive Random Forest.
def predecir_arf(df: pd.DataFrame):

    modelo = cargar_pickle(
        ARF_MODEL_PATH,
        "modelo Adaptive Random Forest"
    )

    scaler = cargar_pickle(
        ARF_SCALER_PATH,
        "escalador Adaptive Random Forest"
    )

    columnas = cargar_pickle(
        ARF_COLUMNS_PATH,
        "columnas Adaptive Random Forest"
    )

    X = preprocesar_datos(df, columnas)

    try:
        X_scaled = scaler.transform(X)

        X_stream = [
            dict(zip(columnas, row))
            for row in X_scaled
        ]

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Error durante el flujo online."
        )

    resultados = []

    try:
        for i, x in enumerate(X_stream):

            pred = modelo.predict_one(x)

            if pred is None:
                pred = 0

            prob = modelo.predict_proba_one(x).get(1, 0)

            resultados.append(
                formatear_prediccion(
                    i,
                    pred,
                    prob
                )
            )

        return resultados

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible completar "
                "la simulación online."
            )
        )


# Resumen agregado de predicciones.
def resumen_predicciones(predicciones):

    total = len(predicciones)

    fraudes = sum(
        1
        for p in predicciones
        if p["resultado"] == "Fraude"
    )

    return {
        "total_curvas": total,
        "fraudes_detectados": fraudes,
        "consumos_normales": total - fraudes
    }


# Predicción histórica desde CSV.
def realizar_prediccion_historica_desde_csv(
    contenido: bytes,
    nombre_archivo: str,
):

    df = leer_csv_subido(contenido)

    predicciones = predecir_xgboost(df)

    resultado = {
        "mensaje": (
            "Predicción histórica de fraude "
            "realizada correctamente."
        ),
        "tipo_prediccion": "historica",
        "archivo_importado": nombre_archivo,
        "modelo_utilizado": {
            "categoria": "offline",
            "nombre": "xgboost"
        },
        "procesamiento": {
            "total_registros_procesados": len(predicciones)
        },
        "resumen": resumen_predicciones(predicciones),
        "predicciones": predicciones
    }

    return limpiar_para_json(resultado)


# Predicción en tiempo real desde CSV.
async def realizar_prediccion_tiempo_real_desde_csv(
    contenido: bytes,
    nombre_archivo: str,
    intervalo_segundos: float = 0.1
):

    df = leer_csv_subido(contenido)

    modelo = cargar_pickle(
        ARF_MODEL_PATH,
        "modelo Adaptive Random Forest"
    )

    scaler = cargar_pickle(
        ARF_SCALER_PATH,
        "escalador Adaptive Random Forest"
    )

    columnas = cargar_pickle(
        ARF_COLUMNS_PATH,
        "columnas Adaptive Random Forest"
    )

    X = preprocesar_datos(df, columnas)

    try:
        X_scaled = scaler.transform(X)

        X_stream = [
            dict(zip(columnas, row))
            for row in X_scaled
        ]

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Error durante la preparación del flujo online."
        )

    cola = asyncio.Queue()

    resultados = []

    async def productor():

        for indice, curva in enumerate(X_stream):

            await cola.put((indice, curva))

            await asyncio.sleep(intervalo_segundos)

        await cola.put(None)

    async def consumidor():

        while True:

            item = await cola.get()

            if item is None:
                break

            indice, curva = item

            try:
                pred = modelo.predict_one(curva)

                if pred is None:
                    pred = 0

                prob = modelo.predict_proba_one(curva).get(1, 0)

                resultados.append(
                    formatear_prediccion(
                        indice,
                        pred,
                        prob
                    )
                )

            except Exception:

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "No ha sido posible clasificar "
                        "una curva del flujo online."
                    )
                )

    try:
        await asyncio.gather(
            productor(),
            consumidor()
        )

    except HTTPException:
        raise

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible completar "
                "la simulación en tiempo real."
            )
        )

    resultado = {
        "mensaje": (
            "Predicción de fraude en tiempo real "
            "realizada correctamente."
        ),
        "tipo_prediccion": "tiempo_real",
        "archivo_importado": nombre_archivo,
        "modelo_utilizado": {
            "categoria": "online",
            "nombre": "adaptive_random_forest"
        },
        "configuracion_flujo": {
            "tecnologia_cola": "asyncio.Queue",
            "intervalo_segundos": intervalo_segundos,
            "registros_procesados": len(resultados)
        },
        "resumen": resumen_predicciones(resultados),
        "predicciones": resultados
    }

    return limpiar_para_json(resultado)


# Importa un archivo CSV y almacena sus curvas.
def importar_archivo_csv(
    contenido: bytes,
    nombre_archivo: str,
    id_usuario: int
):

    df = leer_csv_subido(contenido)

    id_archivo = guardar_archivo(
        id_usuario=id_usuario,
        nombre_archivo=nombre_archivo
    )

    total_curvas = 0

    try:
        for indice, fila in df.iterrows():

            identificador_curva = (
                f"CURVA_{indice + 1}"
            )

            datos_consumo = fila.to_dict()

            guardar_curva(
                id_archivo=id_archivo,
                identificador_curva=identificador_curva,
                datos_consumo=datos_consumo
            )

            total_curvas += 1

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible procesar "
                "las curvas del archivo."
            )
        )

    return {
        "mensaje": "Archivo importado correctamente.",
        "id_archivo": id_archivo,
        "nombre_archivo": nombre_archivo,
        "total_curvas": total_curvas
    }


# Lista curvas de un archivo.
def listar_curvas_archivo(id_archivo: int):

    curvas = obtener_curvas_archivo(id_archivo)

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


# Obtiene detalle completo de una curva.
def obtener_detalle_curva(id_curva: int):

    curva = obtener_curva_por_id(id_curva)

    if curva is None:

        raise HTTPException(
            status_code=404,
            detail="La curva no existe."
        )

    try:
        datos_consumo = json.loads(
            curva["datos_consumo"]
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible recuperar "
                "los datos de consumo."
            )
        )

    return {
        "id_curva": curva["id_curva"],
        "identificador_curva": (
            curva["identificador_curva"]
        ),
        "datos_consumo": datos_consumo
    }


# Predicción histórica de una curva concreta.
def predecir_curva_historica(id_curva: int):

    curva = obtener_curva_por_id(id_curva)

    if curva is None:

        raise HTTPException(
            status_code=404,
            detail="La curva no existe."
        )

    try:
        datos_consumo = json.loads(
            curva["datos_consumo"]
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible recuperar "
                "los datos de consumo."
            )
        )

    try:
        df = pd.DataFrame([datos_consumo])

        modelo = cargar_pickle(
            XGBOOST_MODEL_PATH,
            "modelo XGBoost"
        )

        columnas = cargar_pickle(
            XGBOOST_COLUMNS_PATH,
            "columnas XGBoost"
        )

        X = preprocesar_datos(
            df,
            columnas,
            limpiar_nombres=True
        )

        prediccion = modelo.predict(X)[0]

        probabilidad = modelo.predict_proba(X)[0][1]

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible realizar "
                "la predicción histórica."
            )
        )

    return {
        "id_curva": curva["id_curva"],
        "identificador_curva": (
            curva["identificador_curva"]
        ),
        "modelo": "xgboost",
        "resultado": (
            "Fraude"
            if int(prediccion) == 1
            else "Normal"
        ),
        "probabilidad_fraude": (
            f"{probabilidad * 100:.2f}%"
        )
    }
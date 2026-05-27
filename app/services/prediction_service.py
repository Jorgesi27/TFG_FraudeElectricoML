import json
import math
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
    obtener_estadisticas_archivo
)

BASE_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = BASE_DIR / "models"

XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
XGBOOST_COLUMNS_PATH = MODELS_DIR / "xgboost_columns.pkl"

ARF_MODEL_PATH = MODELS_DIR / "arf_model.pkl"
ARF_COLUMNS_PATH = MODELS_DIR / "arf_columns.pkl"

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

# Carga global de modelos offline.
XGBOOST_MODEL = cargar_pickle(
    XGBOOST_MODEL_PATH,
    "modelo XGBoost"
)

XGBOOST_COLUMNS = cargar_pickle(
    XGBOOST_COLUMNS_PATH,
    "columnas XGBoost"
)

# Carga global de modelos online.
ARF_MODEL = cargar_pickle(
    ARF_MODEL_PATH,
    "modelo Adaptive Random Forest"
)

ARF_COLUMNS = cargar_pickle(
    ARF_COLUMNS_PATH,
    "columnas Adaptive Random Forest"
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

    # Eliminar columnas no necesarias del conjunto de datos de prueba.

    COLUMNAS_NO_FEATURES = [
    "CONS_NO",
        "FLAG"
    ]

    for col in COLUMNAS_NO_FEATURES:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

# Aplica el mismo preprocesamiento usado en entrenamiento.
def preprocesar_datos(
    df_original: pd.DataFrame,
    columnas_esperadas,
    limpiar_nombres=False
):

    try:
        df = df_original.copy()

        COLUMNAS_NO_FEATURES = [
            "CONS_NO",
            "FLAG"
        ]

        for col in COLUMNAS_NO_FEATURES:
            if col in df.columns:
                df = df.drop(columns=[col])

        # limpiar nombres
        if limpiar_nombres:
            df.columns = df.columns.str.replace(
                r"[\[\]<>\(\)]",
                "",
                regex=True
            )

        # asegurar columnas correctas
        df = df.reindex(
            columns=columnas_esperadas,
            fill_value=0
        )

        try:
            df = df[sorted(
                df.columns,
                key=lambda x: pd.to_datetime(x)
            )]
        except:
            pass

        # convertir a numérico
        df = df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        df = df.fillna(method="ffill", axis=1)
        df = df.fillna(0)

        return df

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

        "probabilidad": float(probabilidad) * 100,

        "probabilidad_fraude": (
            f"{float(probabilidad) * 100:.2f}%"
        )
    }

# Predicción histórica con XGBoost.
def predecir_xgboost(df: pd.DataFrame):

    modelo = XGBOOST_MODEL
    columnas = XGBOOST_COLUMNS

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

# Importa un archivo CSV y almacena sus curvas.
def importar_archivo_csv(
    contenido: bytes,
    nombre_archivo: str,
    id_usuario: int
):

    df = leer_csv_subido(contenido)

    try:
        id_archivo = guardar_archivo(
            id_usuario=id_usuario,
            nombre_archivo=nombre_archivo
        )

    except Exception:
        raise

    total_curvas = 0

    try:

        for indice, fila in df.iterrows():

            identificador_curva = f"CURVA_{indice + 1}"

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

    estadisticas = generar_estadisticas_archivo(
        id_archivo,
        id_usuario
    )

    guardar_estadisticas_archivo(
        id_archivo,
        estadisticas
    )

    return {

        "mensaje":
            "Archivo importado correctamente.",

        "id_archivo":
            id_archivo,

        "nombre_archivo":
            nombre_archivo,

        "total_curvas":
            total_curvas
    }

# Lista curvas de un archivo.
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

# Obtiene detalle completo de una curva.
def obtener_detalle_curva(
    id_curva: int,
    id_usuario: int
):

    curva = obtener_curva_por_id(
        id_curva,
        id_usuario
    )

    if curva is None:

        raise HTTPException(
            status_code=404,
            detail="La curva no existe."
        )

    datos_consumo = curva["datos_consumo"]

    if isinstance(datos_consumo, str):
        datos_consumo = json.loads(datos_consumo)

    labels = list(datos_consumo.keys())

    valores = list(datos_consumo.values())

    return {

        "id_curva":
            curva["id_curva"],

        "identificador_curva":
            curva["identificador_curva"],

        "labels":
            labels,

        "valores":
            valores,

        "datos_consumo":
            datos_consumo
    }

# Predicción histórica de una curva concreta.
def predecir_curva_historica(id_curva: int, id_usuario: int):

    curva = obtener_curva_por_id(id_curva, id_usuario)

    if curva is None:

        raise HTTPException(
            status_code=404,
            detail="La curva no existe."
        )

    datos_consumo = curva["datos_consumo"]

    try:

        if isinstance(datos_consumo, str):
            datos_consumo = json.loads(datos_consumo)

        df = pd.DataFrame([datos_consumo])

        resultado = predecir_xgboost(df)[0]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible realizar "
                "la predicción histórica."
            )
        )

    guardar_prediccion(
        id_curva=id_curva,
        tipo_modelo="xgboost",
        tipo_prediccion="historica",
        resultado_prediccion=(
            1 if resultado["resultado"] == "Fraude"
            else 0
        ),
        probabilidad_fraude=(
            resultado["probabilidad"] / 100
        )
    )

    return {

        "id_curva":
            curva["id_curva"],

        "identificador_curva":
            curva["identificador_curva"],

        "modelo":
            "xgboost",

        "resultado":
            resultado["resultado"],

        "probabilidad_fraude":
            resultado["probabilidad_fraude"]
    }

# Predicción de una curva a través del modelo online
def predecir_curva_tiempo_real(
    id_curva: int,
    id_usuario: int
):

    curva = obtener_curva_por_id(
        id_curva,
        id_usuario
    )

    if curva is None:

        raise HTTPException(
            status_code=404,
            detail="La curva no existe."
        )

    datos_consumo = curva["datos_consumo"]

    try:

        # SOPORTAR JSON STRING

        if isinstance(datos_consumo, str):
            datos_consumo = json.loads(
                datos_consumo
            )

        if isinstance(datos_consumo, str):
            datos_consumo = json.loads(datos_consumo)

        df = pd.DataFrame([datos_consumo])

        modelo = ARF_MODEL

        columnas = ARF_COLUMNS

        # PREPROCESAR

        X = preprocesar_datos(
            df,
            columnas
        )

        x_stream = dict(
            zip(columnas, X.iloc[0])
        )

        # PREDICCION

        prediccion = modelo.predict_one(
            x_stream
        )

        if prediccion is None:
            prediccion = 0

        probabilidad = (
            modelo.predict_proba_one(
                x_stream
            ).get(1, 0)
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible realizar "
                "la predicción en tiempo real."
            )
        )

    # GUARDAR EN BD

    guardar_prediccion(
        id_curva=id_curva,
        tipo_modelo="adaptive_random_forest",
        tipo_prediccion="tiempo_real",
        resultado_prediccion=int(prediccion),
        probabilidad_fraude=float(probabilidad)
    )

    return {

        "id_curva":
            curva["id_curva"],

        "identificador_curva":
            curva["identificador_curva"],

        "modelo":
            "adaptive_random_forest",

        "resultado": (
            "Fraude"
            if int(prediccion) == 1
            else "Normal"
        ),

        "probabilidad_fraude":
            f"{probabilidad * 100:.2f}%"
    }

# Generar las estadísticas de un archivo.
def generar_estadisticas_archivo(
    id_archivo: int,
    id_usuario: int
):
    
    # CACHE BD

    estadisticas_guardadas = obtener_estadisticas_archivo(
        id_archivo,
        id_usuario
    )

    if estadisticas_guardadas:
        return estadisticas_guardadas

    curvas = obtener_curvas_archivo(
        id_archivo,
        id_usuario
    )

    if not curvas:

        raise HTTPException(
            status_code=404,
            detail="No existen curvas."
        )

    total = 0
    fraudes = 0
    normales = 0

    probabilidades = []

    consumos_medios = []

    curvas_estadisticas = []

    for curva_resumen in curvas:

        curva = obtener_curva_por_id(
            curva_resumen["id_curva"],
            id_usuario
        )

        datos = curva["datos_consumo"]

        if isinstance(datos, str):
            datos = json.loads(datos)

        if isinstance(datos, str):
            datos = json.loads(datos)

        df = pd.DataFrame([datos])

        columnas_numericas = df.select_dtypes(
            include=["number"]
        ).columns

        if len(columnas_numericas) == 0:
            continue

        consumo_medio = round(
            df[columnas_numericas].mean(axis=1).iloc[0],
            2
        )

        consumos_medios.append(consumo_medio)

        resultado = predecir_xgboost(df)[0]

        total += 1

        probabilidad = resultado["probabilidad"]

        probabilidades.append(
            probabilidad
        )

        curvas_estadisticas.append({

            "curva":
                curva["identificador_curva"],

            "consumo":
                consumo_medio,

            "probabilidad":
                probabilidad
        })

        if resultado["resultado"] == "Fraude":
            fraudes += 1
        else:
            normales += 1

    # TOP CONSUMOS

    top_consumos = sorted(

        curvas_estadisticas,

        key=lambda x: x["consumo"],

        reverse=True

    )[:10]

    # TOP RIESGO

    top_riesgo = sorted(

        curvas_estadisticas,

        key=lambda x: x["probabilidad"],

        reverse=True

    )[:10]

    porcentaje_fraudes = round(
        (fraudes / total) * 100,
        2
    )

    porcentaje_normales = round(
        (normales / total) * 100,
        2
    )

    estadisticas = {

        "id_archivo": id_archivo,

        "total_curvas": total,

        "fraudes": fraudes,

        "normales": normales,

        "porcentaje_fraudes":
            porcentaje_fraudes,

        "porcentaje_normales":
            porcentaje_normales,

        "probabilidades":
            probabilidades,

        "consumos_medios":
            consumos_medios,

        "top_consumos":
            top_consumos,

        "top_riesgo":
            top_riesgo
    }

    estadisticas = limpiar_para_json(estadisticas)

    guardar_estadisticas_archivo(
        id_archivo,
        estadisticas
    )

    return estadisticas


#Predicción online
# Predicción online progresiva
def predecir_stream(valores):

    try:

        modelo = ARF_MODEL

        columnas = ARF_COLUMNS

        if not valores:

            raise HTTPException(
                status_code=400,
                detail="No hay valores para predecir."
            )

        # CREAR STREAM PARCIAL
        datos = {}

        for i, valor in enumerate(valores):

            if i >= len(columnas):
                break

            try:
                datos[columnas[i]] = float(valor)

            except:
                datos[columnas[i]] = 0.0

        if valores:

            inicio = len(valores)

            ultimo_valor = float(valores[-1])

            for i in range(inicio, len(columnas)):
                datos[columnas[i]] = ultimo_valor

        # DATAFRAME
        df = pd.DataFrame([datos])

        # PREPROCESAR
        X = preprocesar_datos(
            df,
            columnas
        )

        x_stream = dict(
            zip(columnas, X.iloc[0])
        )

        # PREDICCIÓN
        prediccion = modelo.predict_one(
            x_stream
        )

        if prediccion is None:
            prediccion = 0

        # PROBABILIDADES
        probabilidades = modelo.predict_proba_one(
            x_stream
        )

        probabilidad = probabilidades.get(
            True,
            probabilidades.get(1, 0)
        )

        return {

            "resultado": (
                "Fraude"
                if int(prediccion) == 1
                else "Normal"
            ),

            "probabilidad": float(probabilidad),

            "probabilidad_fraude":
                f"{probabilidad * 100:.2f}%"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
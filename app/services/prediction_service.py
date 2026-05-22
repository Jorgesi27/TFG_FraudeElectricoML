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


# ========================================
# CARGA GLOBAL MODELOS STREAMING
# ========================================

ARF_SCALER = cargar_pickle(
    ARF_SCALER_PATH,
    "escalador Adaptive Random Forest"
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

        "probabilidad": float(probabilidad) * 100,

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

    id_archivo = guardar_archivo(
        id_usuario=id_usuario,
        nombre_archivo=nombre_archivo
    )

    total_curvas = 0

    try:

        columna_tiempo = detectar_columna_temporal(df)

        for indice, fila in df.iterrows():

            identificador_curva = f"CURVA_{indice + 1}"

            fila_dict = fila.to_dict()

            timestamps = []
            values = []

            for columna, valor in fila_dict.items():

                if columna == columna_tiempo:
                    continue

                timestamps.append(columna)
                values.append(valor)

            datos_consumo = {
                "timestamps": timestamps,
                "values": values
            }

            guardar_curva(
                id_archivo=id_archivo,
                identificador_curva=identificador_curva,
                datos_consumo=datos_consumo
            )

            total_curvas += 1

        # ========================================
        # GENERAR ESTADISTICAS AL IMPORTAR
        # ========================================

    except Exception as e:

        print("ERROR IMPORTACION:", e)

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
def obtener_detalle_curva(id_curva: int, id_usuario: int):

    curva = obtener_curva_por_id(id_curva, id_usuario)

    if curva is None:

        raise HTTPException(
            status_code=404,
            detail="La curva no existe."
        )

    datos_consumo = curva["datos_consumo"]

    # ====================================
    # SOPORTAR STRING JSON
    # ====================================

    if isinstance(datos_consumo, str):
        datos_consumo = json.loads(datos_consumo)

    # ====================================
    # OBTENER LABELS Y VALORES
    # ====================================

    labels = datos_consumo.get("timestamps")

    if labels is None:
        labels = datos_consumo.get("labels", [])

    valores = datos_consumo.get("values")

    if valores is None:
        valores = datos_consumo.get("valores", [])

    return {

        "id_curva":
            curva["id_curva"],

        "identificador_curva":
            curva["identificador_curva"],

        "labels":
            labels,

        "valores":
            valores,

        "datos_consumo": {
            "timestamps": labels,
            "values": valores
        }
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

        datos_modelo = {}

        for t, v in zip(
            datos_consumo["timestamps"],
            datos_consumo["values"]
        ):

            try:
                datos_modelo[str(t)] = float(v)

            except:
                datos_modelo[str(t)] = 0.0

        df = pd.DataFrame([datos_modelo])

        resultado = predecir_xgboost(df)[0]

    except Exception as e:

        print("ERROR HISTORICA:", e)

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

        # ====================================
        # SOPORTAR JSON STRING
        # ====================================

        if isinstance(datos_consumo, str):
            datos_consumo = json.loads(
                datos_consumo
            )

        # ====================================
        # COMPATIBILIDAD FORMATOS
        # ====================================

        timestamps = (
            datos_consumo.get("timestamps")
            or datos_consumo.get("labels")
        )

        values = (
            datos_consumo.get("values")
            or datos_consumo.get("valores")
        )

        if not timestamps or not values:

            raise HTTPException(
                status_code=400,
                detail=(
                    "La curva no contiene "
                    "datos válidos."
                )
            )

        # ====================================
        # CREAR DATAFRAME
        # ====================================

        datos_modelo = {}

        for t, v in zip(
            datos_consumo["timestamps"],
            datos_consumo["values"]
        ):

            try:
                datos_modelo[str(t)] = float(v)

            except:
                datos_modelo[str(t)] = 0.0

        df = pd.DataFrame([datos_modelo])

        # ====================================
        # CARGAR MODELO
        # ====================================

        modelo = cargar_pickle(
            ARF_MODEL_PATH,
            "modelo Adaptive Random Forest"
        )

        scaler = ARF_SCALER

        columnas = ARF_COLUMNS

        # ====================================
        # PREPROCESAR
        # ====================================

        X = preprocesar_datos(
            df,
            columnas
        )

        X_scaled = scaler.transform(X)

        x_stream = dict(
            zip(columnas, X_scaled[0])
        )

        # ====================================
        # PREDICCION
        # ====================================

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

        print("ERROR TIEMPO REAL:", e)

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible realizar "
                "la predicción en tiempo real."
            )
        )

    # ====================================
    # GUARDAR EN BD
    # ====================================

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


def generar_estadisticas_archivo(
    id_archivo: int,
    id_usuario: int
):
    
    # ========================================
    # CACHE BD
    # ========================================

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

        valores = datos.get("values", [])

        if not valores:
            continue

        # convertir a float y eliminar inválidos
        valores_numericos = []

        for v in valores:

            try:
                valores_numericos.append(float(v))

            except (ValueError, TypeError):
                continue

        if not valores_numericos:
            continue

        consumo_medio = round(
            sum(valores_numericos) / len(valores_numericos),
            2
        )

        consumos_medios.append(
            consumo_medio
        )

        datos_modelo = dict(
            zip(
                datos["timestamps"],
                valores_numericos
            )
        )

        df = pd.DataFrame([datos_modelo])

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

    # ====================================
    # TOP CONSUMOS
    # ====================================

    top_consumos = sorted(

        curvas_estadisticas,

        key=lambda x: x["consumo"],

        reverse=True

    )[:10]

    # ====================================
    # TOP RIESGO
    # ====================================

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

    guardar_estadisticas_archivo(
        id_archivo,
        estadisticas
    )

    return estadisticas


def predecir_stream(valores):

    try:

        # =====================================
        # RECARGAR MODELO EN CADA REQUEST
        # =====================================

        modelo = cargar_pickle(
            ARF_MODEL_PATH,
            "modelo Adaptive Random Forest"
        )

        scaler = ARF_SCALER

        columnas = ARF_COLUMNS

        # DEBUG
        print("COLUMNAS:")
        print(columnas[:10])

        # =====================================
        # CREAR DATOS STREAM
        # =====================================

        datos = {}

        # usar solo valores disponibles
        for i, valor in enumerate(valores):

            if i >= len(columnas):
                break

            datos[columnas[i]] = float(valor)

        # =====================================
        # RELLENO PARCIAL
        # SOLO 5 COLUMNAS EXTRA
        # =====================================

        if valores:

            ultimo = float(valores[-1])

            inicio = len(valores)

            fin = min(
                inicio + 5,
                len(columnas)
            )

            for i in range(inicio, fin):

                datos[columnas[i]] = ultimo

        # =====================================
        # DATAFRAME
        # =====================================

        df = pd.DataFrame([datos])

        # DEBUG
        print("--------------------------------")
        print("INPUT DF:")
        print(df.head())
        print("--------------------------------")

        # =====================================
        # PREPROCESAR
        # =====================================

        X = preprocesar_datos(
            df,
            columnas
        )

        X_scaled = scaler.transform(X)

        x_stream = dict(
            zip(columnas, X_scaled[0])
        )

        # DEBUG
        print("--------------------------------")
        print("STREAM FEATURES:")
        print(list(x_stream.items())[:10])
        print("--------------------------------")

        # =====================================
        # PREDICCION
        # =====================================

        prediccion = modelo.predict_one(
            x_stream
        )

        if prediccion is None:
            prediccion = 0

        probabilidades = modelo.predict_proba_one(
            x_stream
        )

        probabilidad = probabilidades.get(1, 0)

        # DEBUG
        print("--------------------------------")
        print("VALORES:", valores)
        print("PREDICCION:", prediccion)
        print("PROBABILIDAD:", probabilidad)
        print("--------------------------------")

        return {

            "resultado": (
                "Fraude"
                if int(prediccion) == 1
                else "Normal"
            ),

            "probabilidad_fraude":
                f"{probabilidad * 100:.2f}%"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    

def detectar_columna_temporal(df):

    posibles = [
        "timestamp",
        "datetime",
        "date",
        "fecha",
        "time",
        "hora"
    ]

    for col in df.columns:

        if col.lower() in posibles:
            return col

    return None
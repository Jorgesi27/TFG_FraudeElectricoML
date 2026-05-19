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
    guardar_prediccion
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

    scaler = ARF_SCALER

    columnas = ARF_COLUMNS

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

    scaler = ARF_SCALER

    columnas = ARF_COLUMNS

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

    return {
        "id_curva": curva["id_curva"],
        "identificador_curva": (
            curva["identificador_curva"]
        ),
        "datos_consumo": datos_consumo
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
    
    guardar_prediccion(
        id_curva=id_curva,
        tipo_modelo="xgboost",
        tipo_prediccion="historica",
        resultado_prediccion=int(prediccion),
        probabilidad_fraude=float(probabilidad)
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


def predecir_curva_tiempo_real(id_curva: int, id_usuario: int):

    curva = obtener_curva_por_id(id_curva, id_usuario)

    if curva is None:

        raise HTTPException(
            status_code=404,
            detail="La curva no existe."
        )

    datos_consumo = curva["datos_consumo"]

    try:

        df = pd.DataFrame([datos_consumo])

        modelo = cargar_pickle(
            ARF_MODEL_PATH,
            "modelo Adaptive Random Forest"
        )

        scaler = ARF_SCALER

        columnas = ARF_COLUMNS

        X = preprocesar_datos(
            df,
            columnas
        )

        X_scaled = scaler.transform(X)

        x_stream = dict(
            zip(columnas, X_scaled[0])
        )

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

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible realizar "
                "la predicción en tiempo real."
            )
        )

    # =========================
    # GUARDAR EN BD
    # =========================

    guardar_prediccion(
        id_curva=id_curva,
        tipo_modelo="adaptive_random_forest",
        tipo_prediccion="tiempo_real",
        resultado_prediccion=int(prediccion),
        probabilidad_fraude=float(probabilidad)
    )

    return {

        "id_curva": curva["id_curva"],

        "identificador_curva": (
            curva["identificador_curva"]
        ),

        "modelo": "adaptive_random_forest",

        "resultado": (
            "Fraude"
            if int(prediccion) == 1
            else "Normal"
        ),

        "probabilidad_fraude": (
            f"{probabilidad * 100:.2f}%"
        )
    }


def generar_estadisticas_archivo(
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
            detail="No existen curvas."
        )

    total = 0
    fraudes = 0
    normales = 0

    probabilidades = []

    consumos_medios = []

    top_curvas = []

    evolucion_consumo = []

    for curva_resumen in curvas:

        curva = obtener_curva_por_id(
            curva_resumen["id_curva"],
            id_usuario
        )

        datos = curva["datos_consumo"]

        if isinstance(datos, str):
            datos = json.loads(datos)

        valores = [

            v for v in datos.values()

            if isinstance(v, (int, float))
        ]

        if not valores:
            continue

        consumo_medio = round(
            sum(valores) / len(valores),
            2
        )

        consumos_medios.append(
            consumo_medio
        )

        if not evolucion_consumo:

            evolucion_consumo = [
                0
            ] * len(valores)

        for i, valor in enumerate(valores):

            evolucion_consumo[i] += valor

        df = pd.DataFrame([datos])

        resultado = predecir_xgboost(df)[0]

        total += 1

        probabilidad = float(

            resultado[
                "probabilidad_fraude"
            ].replace("%", "")
        )

        probabilidades.append(
            probabilidad
        )

        top_curvas.append({

            "curva":
                curva["identificador_curva"],

            "probabilidad":
                probabilidad
        })

        if resultado["resultado"] == "Fraude":
            fraudes += 1
        else:
            normales += 1

    evolucion_consumo = [

        round(v / total, 2)

        for v in evolucion_consumo
    ]

    top_curvas = sorted(

        top_curvas,

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

    return {

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

        "top_curvas":
            top_curvas,

        "evolucion_consumo":
            evolucion_consumo
    }


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
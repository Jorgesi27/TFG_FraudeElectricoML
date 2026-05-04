import json
import math
import pickle
from io import BytesIO
from pathlib import Path
import asyncio
import pandas as pd
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
USER_PREDICTIONS_DIR = RESULTS_DIR / "predicciones_usuarios"

XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
XGBOOST_COLUMNS_PATH = MODELS_DIR / "xgboost_columns.pkl"

ARF_MODEL_PATH = MODELS_DIR / "arf_model.pkl"
ARF_SCALER_PATH = MODELS_DIR / "arf_scaler.pkl"
ARF_COLUMNS_PATH = MODELS_DIR / "arf_columns.pkl"

RESULTS_FILE = RESULTS_DIR / "model_results_final.csv"

ROC_PATHS = {
    "xgboost": RESULTS_DIR / "roc_xgboost.png",
    "adaptive_random_forest": RESULTS_DIR / "roc_arf.png"
}

XGBOOST_IMPORTANCE_PATH = RESULTS_DIR / "xgboost_feature_importance.csv"


# Convierte valores no serializables en JSON, como NaN o infinito, a None, ya que algunas métricas de los modelos online pueden no estar disponibles.
# FastAPI no puede devolver NaN en una respuesta JSON estándar.
def limpiar_para_json(obj):
    if isinstance(obj, dict):
        return {k: limpiar_para_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [limpiar_para_json(v) for v in obj]

    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None

    return obj


# Construye la ruta del fichero donde se almacena la última predicción de un usuario. 
# El resultado se separa por usuario y por tipo de predicción.
def obtener_ruta_prediccion_usuario(nombre_usuario: str, tipo_prediccion: str = "ultima"):
    USER_PREDICTIONS_DIR.mkdir(exist_ok=True)

    nombre_seguro = nombre_usuario.replace("/", "_").replace("\\", "_")
    tipo_seguro = tipo_prediccion.replace("/", "_").replace("\\", "_")

    return USER_PREDICTIONS_DIR / f"{nombre_seguro}_{tipo_seguro}_last_prediction.json"


# Carga modelos, escaladores o listas de columnas previamente guardados en disco.
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


# Lee el archivo CSV importado por el operador. 
# También, se eliminan columnas de índice generadas automáticamente al exportar CSV, ya que no forman parte de las variables predictoras usadas por los modelos.
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

    columnas_indice = [c for c in df.columns if c.lower().startswith("unnamed")]

    if columnas_indice:
        df = df.drop(columns=columnas_indice)

    return df


# Aplica a los datos importados el mismo preprocesamiento usado durante el entrenamiento. 
# Además, se elimina la etiqueta real si aparece, se codifica la variable categórica Class y se alinean las columnas con las guardadas durante el entrenamiento.
def preprocesar_datos(df_original: pd.DataFrame, columnas_esperadas, limpiar_nombres=False):
    try:
        df = df_original.copy()

        # Si el CSV procede del dataset original puede contener la etiqueta real. 
        # En la aplicación operativa se elimina porque el objetivo es predecirla.
        if "theft" in df.columns:
            df = df.drop(columns=["theft"])

        # Se reproduce el one-hot encoding utilizado en entrenamiento para que las categorías se representen como variables numéricas compatibles con los modelos.
        if "Class" in df.columns:
            df = pd.get_dummies(df, columns=["Class"])

        # XGBoost puede dar problemas con ciertos caracteres especiales en los nombres de columnas. 
        # Por eso se aplica la misma limpieza usada al entrenar el modelo offline.
        if limpiar_nombres:
            df.columns = df.columns.str.replace(r"[\[\]<>\(\)]", "", regex=True)

        # La alineación garantiza que el CSV importado tenga exactamente las columnas esperadas.
        # Si una categoría no aparece en el nuevo archivo, se crea con valor 0.
        df = df.reindex(columns=columnas_esperadas, fill_value=0)

        # Los modelos requieren valores numéricos. 
        # Si alguna columna no puede convertirse, se considera que el archivo no es compatible con el formato esperado.
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


# Normaliza el formato de salida de las predicciones de ambos modelos.
def formatear_prediccion(indice: int, prediccion, probabilidad: float):
    pred_int = int(prediccion)

    return {
        "indice": int(indice),
        "resultado": "Fraude" if pred_int == 1 else "Normal",
        "probabilidad_fraude": f"{float(probabilidad) * 100:.2f}%"
    }


# Realiza la predicción con el modelo offline seleccionado: XGBoost. 
# Este modelo procesa el conjunto completo de curvas de consumo de una sola vez, siguiendo el mismo proceso que todos los modelos offline.
def predecir_xgboost(df: pd.DataFrame):
    modelo = cargar_pickle(XGBOOST_MODEL_PATH, "modelo XGBoost")
    columnas = cargar_pickle(XGBOOST_COLUMNS_PATH, "columnas XGBoost")

    X = preprocesar_datos(df, columnas, limpiar_nombres=True)

    try:
        predicciones = modelo.predict(X)
        probabilidades = modelo.predict_proba(X)[:, 1]
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible generar predicciones con XGBoost."
        )

    return [
        formatear_prediccion(i, pred, probabilidades[i])
        for i, pred in enumerate(predicciones)
    ]


# Realiza la predicción con el modelo online seleccionado: Adaptive Random Forest.
# Aunque los datos llegan al prototipo mediante un CSV, el modelo online se ejecuta registro a registro. 
# De esta forma se simula un flujo secuencial de curvas de consumo.
def predecir_arf(df: pd.DataFrame):
    modelo = cargar_pickle(ARF_MODEL_PATH, "modelo Adaptive Random Forest")
    scaler = cargar_pickle(ARF_SCALER_PATH, "escalador Adaptive Random Forest")
    columnas = cargar_pickle(ARF_COLUMNS_PATH, "columnas Adaptive Random Forest")

    X = preprocesar_datos(df, columnas)

    try:
        # Se aplica el mismo escalador ajustado durante el entrenamiento online. 
        # Esto evita inconsistencias entre la escala de los datos de entrenamiento y los datos importados.
        X_scaled = scaler.transform(X)

        # River, biblioteca usada en este modelo, trabaja con observaciones individuales en forma de diccionario.
        # Por eso cada fila se transforma al formato esperado por la función predict_one.
        X_stream = [dict(zip(columnas, row)) for row in X_scaled]
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Error durante la simulación del flujo online."
        )

    resultados = []

    try:
        for i, x in enumerate(X_stream):
            pred = modelo.predict_one(x)

            # Si el modelo no devuelve predicción para algún registro, se adopta una salida conservadora para evitar que una respuesta nula rompa el flujo completo de predicción.
            if pred is None:
                pred = 0

            prob = modelo.predict_proba_one(x).get(1, 0)
            resultados.append(formatear_prediccion(i, pred, prob))

        return resultados

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible completar la simulación del flujo online."
        )


# Calcula un resumen agregado de las predicciones obtenidas.
def resumen_predicciones(predicciones):
    total = len(predicciones)
    fraudes = sum(1 for p in predicciones if p["resultado"] == "Fraude")

    return {
        "total_curvas": total,
        "fraudes_detectados": fraudes,
        "consumos_normales": total - fraudes
    }


# Recupera las métricas experimentales de los modelos utilizados por el sistema. 
# El parámetro "modelos" permite devolver solo las métricas asociadas al modelo usado en cada tipo de predicción.
def cargar_metricas_modelos(modelos=None):
    if not RESULTS_FILE.exists():
        return None

    try:
        df = pd.read_csv(RESULTS_FILE)
        df = df.drop_duplicates(subset=["model"], keep="last")

        if modelos is not None:
            df = df[df["model"].isin(modelos)]
        else:
            df = df[df["model"].isin(["xgboost", "adaptive_random_forest"])]

        if df.empty:
            return None

        df = df.replace([float("inf"), float("-inf")], None)
        df = df.where(pd.notnull(df), None)

        return limpiar_para_json(df.to_dict(orient="records"))

    except Exception:
        return None


# Obtiene las variables más importantes del modelo XGBoost.
# Esta información solo se muestra para XGBoost porque en la fase experimental se calculó y almacenó su importancia de variables. 
# Para Adaptive Random Forest no se generó una medida equivalente dentro del alcance del prototipo.
def obtener_importancia_xgboost(top_n: int = 10):
    if not XGBOOST_IMPORTANCE_PATH.exists():
        return None

    try:
        df = pd.read_csv(XGBOOST_IMPORTANCE_PATH).head(top_n).copy()
        df["importance"] = df["importance"].astype(float) * 100

        resultado = [
            {
                "variable": row["feature"],
                "importancia": f"{row['importance']:.2f}%"
            }
            for _, row in df.iterrows()
        ]

        return limpiar_para_json(resultado)

    except Exception:
        return None


# Devuelve información complementaria asociada al modelo utilizado.
# Si la predicción es histórica, solo se devuelven datos de XGBoost. 
# Si la predicción es en tiempo real, solo se devuelven datos de Adaptive Random Forest.
def obtener_datos_complementarios(modelo: str = "todos"):
    if modelo == "xgboost":
        modelos_metricas = ["xgboost"]
    elif modelo == "adaptive_random_forest":
        modelos_metricas = ["adaptive_random_forest"]
    else:
        modelos_metricas = ["xgboost", "adaptive_random_forest"]

    datos = {
        "descripcion": (
            "Información complementaria asociada al modelo utilizado en la predicción. "
            "Incluye métricas de evaluación, disponibilidad de curva ROC e importancia de variables cuando procede."
        ),
        "metricas_modelos": cargar_metricas_modelos(modelos_metricas),
        "curvas_roc": {
            "descripcion_general": (
                "La curva ROC muestra la relación entre la tasa de verdaderos positivos "
                "y la tasa de falsos positivos. Cuanto más cerca esté la curva de la esquina "
                "superior izquierda, mejor es el rendimiento del modelo. La diagonal representa "
                "un comportamiento equivalente a una clasificación aleatoria."
            )
        },
        "importancia_variables": {}
    }

    if modelo in ["xgboost", "todos"]:
        importancia_xgboost = obtener_importancia_xgboost(top_n=10)

        datos["curvas_roc"]["xgboost"] = {
            "disponible": ROC_PATHS["xgboost"].exists(),
            "descripcion": "Curva ROC disponible para el modelo XGBoost.",
            "visualizacion": (
                "Ejecutar el endpoint /api/operador/curva-roc seleccionando el modelo xgboost."
                if ROC_PATHS["xgboost"].exists()
                else "Curva ROC no disponible."
            )
        }

        datos["importancia_variables"]["xgboost"] = {
            "disponible": importancia_xgboost is not None,
            "descripcion": "Top 10 variables más importantes según XGBoost.",
            "valores": importancia_xgboost
        }

    if modelo in ["adaptive_random_forest", "todos"]:
        datos["curvas_roc"]["adaptive_random_forest"] = {
            "disponible": ROC_PATHS["adaptive_random_forest"].exists(),
            "descripcion": "Curva ROC disponible para el modelo Adaptive Random Forest.",
            "visualizacion": (
                "Ejecutar el endpoint /api/operador/curva-roc seleccionando el modelo adaptive_random_forest."
                if ROC_PATHS["adaptive_random_forest"].exists()
                else "Curva ROC no disponible."
            )
        }

        datos["importancia_variables"]["adaptive_random_forest"] = {
            "disponible": False,
            "descripcion": "Importancia de variables no disponible para este modelo en el prototipo.",
            "valores": None
        }

    return limpiar_para_json(datos)


# Almacena el último resultado de predicción para una futura comprobación.
def guardar_ultimo_resultado(resultado, nombre_usuario: str, tipo_prediccion: str):
    ruta_prediccion = obtener_ruta_prediccion_usuario(
        nombre_usuario=nombre_usuario,
        tipo_prediccion=tipo_prediccion
    )

    try:
        with open(ruta_prediccion, "w", encoding="utf-8") as f:
            json.dump(limpiar_para_json(resultado), f, ensure_ascii=False, indent=4)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="La predicción se ha realizado, pero no ha sido posible guardar el último resultado."
        )


# Función auxiliar que comprueba si el usuario ya ha realizado al menos una predicción.
def comprobar_prediccion_previa(nombre_usuario: str):
    ruta_prediccion = obtener_ruta_prediccion_usuario(nombre_usuario)

    if not ruta_prediccion.exists():
        raise HTTPException(
            status_code=404,
            detail="No existe ninguna predicción previa para este usuario. Realice primero una predicción de fraude."
        )


# Función auxiliar para recuperar la última predicción almacenada para un usuario y modalidad concreta.
def cargar_ultimo_resultado(
    nombre_usuario: str,
    tipo_prediccion: str,
    incluir_datos_complementarios: bool = False
):
    comprobar_prediccion_por_tipo(nombre_usuario, tipo_prediccion)

    ruta_prediccion = obtener_ruta_prediccion_usuario(
        nombre_usuario=nombre_usuario,
        tipo_prediccion=tipo_prediccion
    )

    try:
        with open(ruta_prediccion, "r", encoding="utf-8") as f:
            resultado = json.load(f)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible recuperar la última predicción."
        )

    if incluir_datos_complementarios:
        modelo = resultado.get("modelo_utilizado", {}).get("nombre", "todos")
        resultado["datos_complementarios"] = obtener_datos_complementarios(modelo)

    return limpiar_para_json(resultado)


# Ejecuta la predicción histórica de fraude utilizando el modelo XGBoost.
def realizar_prediccion_historica_desde_csv(
    contenido: bytes,
    nombre_archivo: str,
    nombre_usuario: str,
    limite_registros: int = 1000,
    limite_predicciones_mostradas: int = 1000,
    incluir_datos_complementarios: bool = False
):
    df = leer_csv_subido(contenido)

    total_registros_archivo = len(df)
    df = df.head(limite_registros).copy()

    predicciones_xgboost = predecir_xgboost(df)
    predicciones_mostradas = predicciones_xgboost[:limite_predicciones_mostradas]

    resultado = {
        "mensaje": "Predicción histórica de fraude realizada correctamente.",
        "tipo_prediccion": "historica",
        "archivo_importado": nombre_archivo,
        "modelo_utilizado": {
            "categoria": "offline",
            "nombre": "xgboost"
        },
        "procesamiento": {
            "total_registros_archivo": total_registros_archivo,
            "registros_procesados": len(predicciones_xgboost),
            "predicciones_mostradas": len(predicciones_mostradas),
            "limite_registros": limite_registros,
            "limite_predicciones_mostradas": limite_predicciones_mostradas
        },
        "resumen": resumen_predicciones(predicciones_xgboost),
        "predicciones": predicciones_mostradas
    }

    resultado_base = limpiar_para_json(resultado)
    guardar_ultimo_resultado(
        resultado=resultado_base,
        nombre_usuario=nombre_usuario,
        tipo_prediccion="historica"
    )

    respuesta = resultado_base.copy()

    if incluir_datos_complementarios:
        respuesta["datos_complementarios"] = obtener_datos_complementarios("xgboost")

    return limpiar_para_json(respuesta)


# Ejecuta la predicción de fraude en tiempo real utilizando Adaptive Random Forest.
# El flujo simula un entorno online mediante una cola interna. 
# Un productor introduce progresivamente las curvas de consumo en la cola y un consumidor las procesa una a una.
async def realizar_prediccion_tiempo_real_desde_csv(
    contenido: bytes,
    nombre_archivo: str,
    nombre_usuario: str,
    intervalo_segundos: float = 0.1,
    limite_registros: int = 1000,
    limite_predicciones_mostradas: int = 1000,
    incluir_datos_complementarios: bool = False
):
    df = leer_csv_subido(contenido)

    total_registros_archivo = len(df)

    if limite_registros is not None:
        df = df.head(limite_registros).copy()

    modelo = cargar_pickle(ARF_MODEL_PATH, "modelo Adaptive Random Forest")
    scaler = cargar_pickle(ARF_SCALER_PATH, "escalador Adaptive Random Forest")
    columnas = cargar_pickle(ARF_COLUMNS_PATH, "columnas Adaptive Random Forest")

    X = preprocesar_datos(df, columnas)

    try:
        X_scaled = scaler.transform(X)
        X_stream = [dict(zip(columnas, row)) for row in X_scaled]
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Error durante la preparación del flujo online."
        )

    cola = asyncio.Queue()
    resultados = []

    # Simula la llegada progresiva de curvas de consumo.
    # Cada registro se introduce en la cola con una espera entre envíos para representar la recepción periódica de datos en un escenario online.
    async def productor():
        for indice, curva in enumerate(X_stream):
            await cola.put((indice, curva))
            await asyncio.sleep(intervalo_segundos)

        await cola.put(None)

    # Consume las curvas de la cola y las clasifica con Adaptive Random Forest.
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
                        indice=indice,
                        prediccion=pred,
                        probabilidad=prob
                    )
                )

            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail="No ha sido posible clasificar una curva del flujo online."
                )

    try:
        await asyncio.gather(productor(), consumidor())
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible completar la simulación del flujo en tiempo real."
        )

    predicciones_mostradas = resultados[:limite_predicciones_mostradas]

    resultado = {
        "mensaje": "Predicción de fraude en tiempo real realizada correctamente.",
        "tipo_prediccion": "tiempo_real",
        "archivo_importado": nombre_archivo,
        "modelo_utilizado": {
            "categoria": "online",
            "nombre": "adaptive_random_forest"
        },
        "configuracion_flujo": {
            "tecnologia_cola": "asyncio.Queue",
            "intervalo_segundos": intervalo_segundos,
            "total_registros_archivo": total_registros_archivo,
            "registros_procesados": len(resultados),
            "predicciones_mostradas": len(predicciones_mostradas),
            "limite_registros": limite_registros,
            "limite_predicciones_mostradas": limite_predicciones_mostradas
        },
        "resumen": resumen_predicciones(resultados),
        "predicciones": predicciones_mostradas
    }

    resultado_base = limpiar_para_json(resultado)

    guardar_ultimo_resultado(
        resultado=resultado_base,
        nombre_usuario=nombre_usuario,
        tipo_prediccion="tiempo_real"
    )

    respuesta = resultado_base.copy()

    if incluir_datos_complementarios:
        respuesta["datos_complementarios"] = obtener_datos_complementarios("adaptive_random_forest")

    return limpiar_para_json(respuesta)

# Función auxiliar que verifica si existe una predicción previa para el usuario y modalidad solicitada.
def comprobar_prediccion_por_tipo(nombre_usuario: str, tipo_prediccion: str):
    ruta_prediccion = obtener_ruta_prediccion_usuario(nombre_usuario, tipo_prediccion)

    if not ruta_prediccion.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No existe ninguna predicción previa de tipo {tipo_prediccion} para este usuario."
        )
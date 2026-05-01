from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()


@router.post(
    "/login",
    summary="Iniciar sesión",
    description="Permite al operador iniciar sesión en el sistema."
)
def iniciar_sesion(usuario: str, password: str):
    # Implementación futura: validación contra base de datos o sistema de autenticación.
    if not usuario or not password:
        raise HTTPException(
            status_code=400,
            detail="El usuario y la contraseña son obligatorios."
        )

    return {
        "mensaje": "Inicio de sesión realizado correctamente.",
        "usuario": usuario,
        "rol": "operador",
        "estado": "autenticado"
    }


@router.post(
    "/prediccion",
    summary="Realizar predicción de fraude",
    description=(
        "Permite importar un archivo CSV con curvas de consumo eléctrico y realizar "
        "una predicción de fraude utilizando los modelos previamente entrenados."
    )
)
async def realizar_prediccion(archivo: UploadFile = File(...)):
    # Implementación futura:
    # 1. Validar CSV
    # 2. Cargar modelos entrenados
    # 3. Preprocesar datos
    # 4. Realizar predicción offline y online
    # 5. Guardar último resultado

    if not archivo.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe tener formato CSV."
        )

    return {
        "mensaje": "Archivo recibido correctamente. La predicción se implementará en la siguiente fase.",
        "archivo": archivo.filename,
        "modelos_previstos": [
            "XGBoost",
            "Adaptive Random Forest"
        ],
        "resultado": "pendiente_implementacion"
    }


@router.get(
    "/ultimo-resultado",
    summary="Consultar último resultado de predicción",
    description="Devuelve el último resultado de predicción generado por el sistema."
)
def consultar_ultimo_resultado():
    # Implementación futura: recuperar último resultado almacenado.
    return {
        "mensaje": "Consulta del último resultado de predicción.",
        "estado": "pendiente_implementacion",
        "ultimo_resultado": None
    }


@router.get(
    "/datos-complementarios",
    summary="Consultar datos complementarios",
    description=(
        "Devuelve información complementaria de los modelos utilizados, como métricas "
        "de evaluación, curvas ROC e importancia de variables cuando esté disponible."
    )
)
def consultar_datos_complementarios():
    # Implementación futura: conectar con resultados, curvas ROC e importancia de variables.
    return {
        "mensaje": "Consulta de datos complementarios.",
        "estado": "pendiente_implementacion",
        "datos_disponibles": {
            "metricas": True,
            "curvas_roc": True,
            "importancia_variables": True
        }
    }
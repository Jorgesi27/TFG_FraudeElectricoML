from enum import Enum
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse

from app.services.prediction_service import (
    realizar_prediccion_desde_csv,
    cargar_ultimo_resultado,
    comprobar_prediccion_previa
)

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"


class OpcionDatosComplementarios(str, Enum):
    no = "no"
    si = "si"


class ModeloCurvaROC(str, Enum):
    xgboost = "xgboost"
    adaptive_random_forest = "adaptive_random_forest"

@router.post(
    "/login",
    summary="Iniciar sesión",
    description="Permite al operador iniciar sesión en el sistema."
)
def iniciar_sesion(usuario: str, password: str):
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
        "una predicción de fraude usando XGBoost como modelo offline y Adaptive Random "
        "Forest como modelo online."
    )
)
async def realizar_prediccion(
    archivo: UploadFile = File(...),
    incluir_datos_complementarios: OpcionDatosComplementarios = Query(
        default=OpcionDatosComplementarios.no,
        description="Indica si se desean incluir datos complementarios en la respuesta."
    )
):
    if not archivo.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="El archivo importado debe tener formato CSV."
        )

    contenido = await archivo.read()

    return realizar_prediccion_desde_csv(
        contenido=contenido,
        nombre_archivo=archivo.filename,
        incluir_datos_complementarios=(
            incluir_datos_complementarios == OpcionDatosComplementarios.si
        )
    )


@router.get(
    "/ultimo-resultado",
    summary="Consultar último resultado de predicción",
    description="Devuelve el último resultado de predicción almacenado por el sistema."
)
def consultar_ultimo_resultado(
    incluir_datos_complementarios: OpcionDatosComplementarios = Query(
        default=OpcionDatosComplementarios.no,
        description="Indica si se desean incluir métricas, curvas ROC e importancia de variables."
    )
):
    return cargar_ultimo_resultado(
        incluir_datos_complementarios=(
            incluir_datos_complementarios == OpcionDatosComplementarios.si
        )
    )


@router.get(
    "/curva-roc",
    summary="Visualizar curva ROC del modelo",
    description=(
        "Devuelve la imagen de la curva ROC asociada al modelo seleccionado. "
        "La curva ROC permite evaluar la capacidad del modelo para distinguir entre consumo normal y fraude. "
        "Cuanto más próxima esté la curva a la esquina superior izquierda, mejor será el rendimiento del modelo. "
        "La diagonal representa un clasificador aleatorio."
    )
)
def ver_curva_roc(
    modelo: ModeloCurvaROC = Query(
        description="Seleccione el modelo cuya curva ROC desea visualizar."
    )
):
    comprobar_prediccion_previa()

    if modelo == ModeloCurvaROC.xgboost:
        ruta = RESULTS_DIR / "roc_xgboost.png"
    elif modelo == ModeloCurvaROC.adaptive_random_forest:
        ruta = RESULTS_DIR / "roc_arf.png"
    else:
        raise HTTPException(
            status_code=404,
            detail="Modelo no reconocido."
        )

    if not ruta.exists():
        raise HTTPException(
            status_code=404,
            detail="La curva ROC solicitada no está disponible."
        )

    return FileResponse(
        path=ruta,
        media_type="image/png",
        filename=ruta.name
    )
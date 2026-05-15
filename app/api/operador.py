from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Form,
    Request
)

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.auth import autenticar_usuario

from app.services.prediction_service import (
    importar_archivo_csv,
    listar_curvas_archivo,
    predecir_curva_historica,
    realizar_prediccion_historica_desde_csv,
    realizar_prediccion_tiempo_real_desde_csv,
    obtener_detalle_curva
)

from app.core.database import (
    obtener_archivos_usuario
)

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]

templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def interfaz(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# Login temporal (sin protección real aún)
@router.post("/login")
def iniciar_sesion(
    usuario: str = Form(...),
    password: str = Form(...)
):

    return autenticar_usuario(
        nombre_usuario=usuario,
        password=password
    )


@router.post("/importar-csv")
async def importar_csv(
    archivo: UploadFile = File(...)
):

    if not archivo.filename.endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser CSV."
        )

    contenido = await archivo.read()

    return importar_archivo_csv(
        contenido=contenido,
        nombre_archivo=archivo.filename,
        id_usuario=1
    )


@router.get("/archivos")
def listar_archivos():

    return obtener_archivos_usuario(1)


@router.get("/curvas/{id_archivo}")
def listar_curvas(id_archivo: int):

    return listar_curvas_archivo(id_archivo)


@router.get("/curva/{id_curva}")
def obtener_curva(id_curva: int):

    return obtener_detalle_curva(id_curva)


@router.post("/prediccion/historica/{id_curva}")
def predecir_historica(id_curva: int):

    return predecir_curva_historica(id_curva)


@router.post("/prediccion/historica")
async def predecir_fraude_historico(
    archivo: UploadFile = File(...)
):

    contenido = await archivo.read()

    return realizar_prediccion_historica_desde_csv(
        contenido=contenido,
        nombre_archivo=archivo.filename
    )


@router.post("/prediccion/tiempo-real")
async def predecir_fraude_tiempo_real(
    archivo: UploadFile = File(...)
):

    contenido = await archivo.read()

    return await realizar_prediccion_tiempo_real_desde_csv(
        contenido=contenido,
        nombre_archivo=archivo.filename
    )
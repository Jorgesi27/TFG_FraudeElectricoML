from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Form,
    Request,
    Depends
)

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from fastapi.security import (
    HTTPAuthorizationCredentials
)

from app.core.auth import (
    autenticar_usuario,
    obtener_operador_actual,
    security,
    cerrar_sesion
)

from app.services.prediction_service import (
    importar_archivo_csv,
    listar_curvas_archivo,
    predecir_curva_historica,
    obtener_detalle_curva,
    generar_estadisticas_archivo,
    predecir_curva_tiempo_real,
    predecir_stream
)

from app.core.database import (
    obtener_archivos_usuario
)

from pydantic import BaseModel
from typing import List

class StreamRequest(BaseModel):
    valores: List[float]


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]

templates = Jinja2Templates(
    directory="app/templates"
)


# ========================================
# PAGINAS HTML
# ========================================

@router.get(
    "/home",
    response_class=HTMLResponse,
    include_in_schema=False
)
def home_page(request: Request):

    return templates.TemplateResponse(
        "home.html",
        {"request": request}
    )


@router.get(
    "/historica",
    response_class=HTMLResponse,
    include_in_schema=False
)
def historica_page(request: Request):

    return templates.TemplateResponse(
        "historica.html",
        {"request": request}
    )


@router.get(
    "/tiempo-real",
    response_class=HTMLResponse,
    include_in_schema=False
)
def tiempo_real_page(request: Request):

    return templates.TemplateResponse(
        "tiempo_real.html",
        {"request": request}
    )


@router.post(
    "/prediccion/stream"
)
def prediccion_stream(
    request: StreamRequest
):

    return predecir_stream(
        request.valores
    )


@router.get(
    "/estadisticas",
    response_class=HTMLResponse,
    include_in_schema=False
)
def estadisticas_page(request: Request):

    return templates.TemplateResponse(
        "estadisticas.html",
        {"request": request}
    )


@router.get(
    "/login-page",
    response_class=HTMLResponse,
    include_in_schema=False
)
def login_page(request: Request):

    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )


# ========================================
# AUTH
# ========================================

@router.post("/login")
def iniciar_sesion(
    usuario: str = Form(...),
    password: str = Form(...)
):

    return autenticar_usuario(
        nombre_usuario=usuario,
        password=password
    )


@router.post("/logout")
def logout(
    credenciales:
    HTTPAuthorizationCredentials =
    Depends(security)
):

    return cerrar_sesion(
        credenciales.credentials
    )


# ========================================
# IMPORTAR CSV
# ========================================

@router.post("/importar-csv")
async def importar_csv(

    usuario_actual: dict = Depends(
        obtener_operador_actual
    ),

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
        id_usuario=usuario_actual["id_usuario"]
    )


# ========================================
# ARCHIVOS
# ========================================

@router.get("/archivos")
def listar_archivos(

    usuario_actual: dict = Depends(
        obtener_operador_actual
    )
):

    return obtener_archivos_usuario(
        usuario_actual["id_usuario"]
    )


# ========================================
# CURVAS
# ========================================

@router.get("/curvas/{id_archivo}")
def listar_curvas(

    id_archivo: int,

    usuario_actual: dict = Depends(
        obtener_operador_actual
    )
):

    return listar_curvas_archivo(
        id_archivo,
        usuario_actual["id_usuario"]
    )


@router.get("/curva/{id_curva}")
def obtener_curva(

    id_curva: int,

    usuario_actual: dict = Depends(
        obtener_operador_actual
    )
):

    return obtener_detalle_curva(
        id_curva,
        usuario_actual["id_usuario"]
    )


# ========================================
# PREDICCION HISTORICA
# ========================================

@router.post(
    "/prediccion/historica/{id_curva}"
)
def predecir_historica(

    id_curva: int,

    usuario_actual: dict = Depends(
        obtener_operador_actual
    )
):

    return predecir_curva_historica(
        id_curva,
        usuario_actual["id_usuario"]
    )


# ========================================
# PREDICCION TIEMPO REAL
# ========================================

@router.post(
    "/prediccion/tiempo-real/{id_curva}"
)
def predecir_tiempo_real(

    id_curva: int,

    usuario_actual: dict = Depends(
        obtener_operador_actual
    )
):

    return predecir_curva_tiempo_real(
        id_curva,
        usuario_actual["id_usuario"]
    )

# ========================================
# ESTADISTICAS
# ========================================

@router.get(
    "/estadisticas/{id_archivo}"
)
def obtener_estadisticas(

    id_archivo: int,

    usuario_actual: dict = Depends(
        obtener_operador_actual
    )
):

    return generar_estadisticas_archivo(
        id_archivo,
        usuario_actual["id_usuario"]
    )
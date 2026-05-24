from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from app.api import operador

# Configuración principal FastAPI.
app = FastAPI(
    title="API de Detección de Fraude Eléctrico",
    description=(
        "Sistema para importar curvas de consumo eléctrico y predecir posibles casos "
        "de fraude mediante modelos de aprendizaje automático."
    ),
    version="1.0.0",
    docs_url="/documentacion",
    redoc_url=None
)

# Archivos estáticos (CSS Y JS).
app.mount("/static", StaticFiles(directory="app/static"), name="static")

#Configuración de plantillas html.
templates = Jinja2Templates(directory="app/templates")

# Registro de rutas del operador.
app.include_router(
    operador.router,
    prefix="/api/operador",
    tags=["Operador"]
)

# Redirección a la página principal.
@app.get("/", include_in_schema=False)
def root():

    return RedirectResponse(
        url="/api/operador/home"
    )
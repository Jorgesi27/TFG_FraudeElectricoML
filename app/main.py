from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.api import operador

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


app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")


app.include_router(
    operador.router,
    prefix="/api/operador",
    tags=["Operador"]
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
from fastapi import FastAPI
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

app.include_router(
    operador.router,
    prefix="/api/operador",
    tags=["Operador"]
)


@app.get("/", include_in_schema=False)
def root():
    return {
        "mensaje": "API de Detección de Fraude Eléctrico",
        "estado": "activa",
        "documentacion": "/documentacion"
    }
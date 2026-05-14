from enum import Enum
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends, Form
from fastapi.responses import FileResponse

from app.core.auth import autenticar_usuario, cerrar_sesion, obtener_operador_actual
from app.services.prediction_service import (
    realizar_prediccion_historica_desde_csv,
    realizar_prediccion_tiempo_real_desde_csv,
    cargar_ultimo_resultado,
    comprobar_prediccion_por_tipo
)

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"

# Enum utilizado para controlar desde Swagger si se desea incluir información adicional.
class OpcionDatosComplementarios(str, Enum):
    no = "no"
    si = "si"


# Modelos disponibles para la visualización de curvas ROC.
# XGBoost para predicción histórica y Adaptive Random Forest para tiempo real.
class ModeloCurvaROC(str, Enum):
    xgboost = "xgboost"
    adaptive_random_forest = "adaptive_random_forest"


# Tipos de predicción disponibles en el sistema.
# Esta separación permite mantener flujos distintos para consumo histórico y tiempo real.
class TipoPrediccion(str, Enum):
    historica = "historica"
    tiempo_real = "tiempo_real"


# Autentica al usuario contra la base de datos y devuelve un token de acceso.
# El token se utiliza posteriormente para proteger los endpoints operativos del sistema, evitando que usuarios no autenticados puedan realizar predicciones o consultar resultados.
@router.post(
    "/login",
    summary="Iniciar sesión",
    description="Permite al operador iniciar sesión en el sistema."
)
def iniciar_sesion(
    usuario: str = Form(
        ...,
        example="usuario",
        description="Nombre de usuario del operador."
    ),
    password: str = Form(
        ...,
        example="********",
        description="Contraseña asociada al usuario."
    )
):
    return autenticar_usuario(
        nombre_usuario=usuario,
        password=password
    )


# Invalida el token activo del usuario autenticado.
# El cierre de sesión se realiza en el servidor, no solo en la interfaz Swagger. Esto impide que el mismo token pueda seguir utilizándose después de cerrar la sesión.
@router.post(
    "/logout",
    summary="Cerrar sesión",
    description="Permite cerrar la sesión activa del operador."
)
def logout(usuario_actual: dict = Depends(obtener_operador_actual)):
    return cerrar_sesion(usuario_actual["token"])


# Ejecuta el flujo de predicción histórica utilizando únicamente XGBoost.
# Este endpoint está orientado al análisis de archivos CSV ya disponibles, por lo que se limita el número de registros procesados y mostrados para evitar respuestas demasiado grandes y mejorar la estabilidad del prototipo.
@router.post(
    "/prediccion/historica",
    summary="Predecir fraude en consumo histórico",
    description=(
        "Permite importar un archivo CSV con curvas de consumo eléctrico históricas "
        "y realizar una predicción de fraude utilizando el modelo XGBoost, seleccionado "
        "como mejor modelo offline durante la fase experimental."
    )
)
async def predecir_fraude_historico(
    archivo: UploadFile = File(...),
    limite_registros: int = Query(
        default=1000,
        ge=1,
        le=10000,
        description="Número máximo de curvas del CSV que se procesarán."
    ),
    limite_predicciones_mostradas: int = Query(
        default=1000,
        ge=1,
        le=1000,
        description="Número máximo de predicciones individuales que se mostrarán en la respuesta."
    ),
    incluir_datos_complementarios: OpcionDatosComplementarios = Query(
        default=OpcionDatosComplementarios.no,
        description="Indica si se desean incluir datos complementarios en la respuesta."
    ),
    usuario_actual: dict = Depends(obtener_operador_actual)
):
    if not archivo.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="El archivo importado debe tener formato CSV."
        )

    contenido = await archivo.read()

    return realizar_prediccion_historica_desde_csv(
        contenido=contenido,
        nombre_archivo=archivo.filename,
        nombre_usuario=usuario_actual["usuario"],
        limite_registros=limite_registros,
        limite_predicciones_mostradas=limite_predicciones_mostradas,
        incluir_datos_complementarios=(
            incluir_datos_complementarios == OpcionDatosComplementarios.si
        )
    )


# Ejecuta el flujo online usando Adaptive Random Forest y una cola interna.
# Aunque el operador importa un CSV, los registros se introducen progresivamente en una cola para simular la llegada de datos en tiempo real. 
@router.post(
    "/prediccion/tiempo-real",
    summary="Predecir fraude en tiempo real",
    description=(
        "Permite simular la llegada de curvas de consumo en tiempo real mediante una cola interna. "
        "El sistema introduce los registros del archivo CSV en una cola cada cierto intervalo de tiempo "
        "y los clasifica progresivamente utilizando Adaptive Random Forest, seleccionado como mejor modelo online."
    )
)
async def predecir_fraude_tiempo_real(
    archivo: UploadFile = File(...),
    intervalo_segundos: float = Query(
        default=0.1,
        ge=0,
        le=5,
        description="Tiempo de espera entre la llegada de cada curva de consumo a la cola."
    ),
    limite_registros: int = Query(
        default=100,
        ge=1,
        le=5000,
        description="Número máximo de curvas del CSV que se procesarán en la simulación."
    ),
    limite_predicciones_mostradas: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Número máximo de predicciones individuales que se mostrarán en la respuesta."
    ),
    incluir_datos_complementarios: OpcionDatosComplementarios = Query(
        default=OpcionDatosComplementarios.no,
        description="Indica si se desean incluir datos complementarios en la respuesta."
    ),
    usuario_actual: dict = Depends(obtener_operador_actual)
):
    if not archivo.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="El archivo importado debe tener formato CSV."
        )

    contenido = await archivo.read()

    return await realizar_prediccion_tiempo_real_desde_csv(
        contenido=contenido,
        nombre_archivo=archivo.filename,
        nombre_usuario=usuario_actual["usuario"],
        intervalo_segundos=intervalo_segundos,
        limite_registros=limite_registros,
        limite_predicciones_mostradas=limite_predicciones_mostradas,
        incluir_datos_complementarios=(
            incluir_datos_complementarios == OpcionDatosComplementarios.si
        )
    )


# Recupera el último resultado del usuario autenticado para el tipo de predicción elegido.
# Los resultados se almacenan separados por usuario y por modalidad de predicción, de forma que un operador no pueda consultar resultados generados por otro usuario.
@router.get(
    "/ultimo-resultado",
    summary="Consultar último resultado de predicción",
    description=(
        "Devuelve el último resultado de predicción almacenado por el sistema. "
        "Permite seleccionar si se desea consultar el último resultado histórico "
        "o el último resultado en tiempo real."
    )
)
def consultar_ultimo_resultado(
    tipo_prediccion: TipoPrediccion = Query(
        default=TipoPrediccion.historica,
        description="Seleccione el tipo de predicción cuyo último resultado desea consultar."
    ),
    incluir_datos_complementarios: OpcionDatosComplementarios = Query(
        default=OpcionDatosComplementarios.no,
        description="Indica si se desean incluir métricas, curvas ROC e importancia de variables."
    ),
    usuario_actual: dict = Depends(obtener_operador_actual)
):
    return cargar_ultimo_resultado(
        nombre_usuario=usuario_actual["usuario"],
        tipo_prediccion=tipo_prediccion.value,
        incluir_datos_complementarios=(
            incluir_datos_complementarios == OpcionDatosComplementarios.si
        )
    )


# Devuelve la curva ROC únicamente si el usuario ha realizado antes la predicción asociada.
# La curva de XGBoost se habilita tras una predicción histórica, mientras que la curva de ARF se habilita tras una predicción en tiempo real.
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
    ),
    usuario_actual: dict = Depends(obtener_operador_actual)
):
    if modelo == ModeloCurvaROC.xgboost:
        comprobar_prediccion_por_tipo(
            nombre_usuario=usuario_actual["usuario"],
            tipo_prediccion="historica"
        )
        ruta = RESULTS_DIR / "roc_xgboost.png"

    elif modelo == ModeloCurvaROC.adaptive_random_forest:
        comprobar_prediccion_por_tipo(
            nombre_usuario=usuario_actual["usuario"],
            tipo_prediccion="tiempo_real"
        )
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
        filename=ruta.name,
        headers={
            # Evita que el navegador muestre una curva ROC almacenada en caché después de cambiar de usuario o cerrar sesión.
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

from app.core.database import obtener_usuario_por_nombre


security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

TOKEN_EXPIRATION_MINUTES = 60
TOKENS_ACTIVOS = {}


# Función para verifciar la contraseña.
def verificar_password(password: str, password_hash: str):
    return pwd_context.verify(password, password_hash)


# Función para autenticar a cada usuario y controlar los accesos no autorizados.
def autenticar_usuario(nombre_usuario: str, password: str):
    usuario = obtener_usuario_por_nombre(nombre_usuario)

    if usuario is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales no válidas."
        )

    if not usuario["activo"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales no válidas."
        )

    if not verificar_password(password, usuario["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales no válidas."
        )

    token = str(uuid4())

    TOKENS_ACTIVOS[token] = {
        "id_usuario": usuario["id_usuario"],
        "usuario": usuario["nombre_usuario"],
        "rol": usuario["rol"],
        "expira": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRATION_MINUTES)
    }

    return {
        "mensaje": "Inicio de sesión realizado correctamente.",
        "usuario": usuario["nombre_usuario"],
        "rol": usuario["rol"],
        "access_token": token,
        "token_type": "bearer",
        "expires_in_minutes": TOKEN_EXPIRATION_MINUTES
    }


# Función para recuperar el usuario actual.
def obtener_usuario_actual(
    credenciales: HTTPAuthorizationCredentials = Depends(security)
):
    token = credenciales.credentials
    sesion = TOKENS_ACTIVOS.get(token)

    if sesion is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No autorizado."
        )

    if datetime.now(timezone.utc) > sesion["expira"]:
        TOKENS_ACTIVOS.pop(token, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No autorizado."
        )

    return {
        "id_usuario": sesion["id_usuario"],
        "usuario": sesion["usuario"],
        "rol": sesion["rol"],
        "token": token
    }


# Función para recuperar el oeprador actual.
def obtener_operador_actual(usuario_actual: dict = Depends(obtener_usuario_actual)):
    if usuario_actual["rol"] not in ["operador", "administrador"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No autorizado."
        )

    return usuario_actual


# Cerrar sesión del usuario.
def cerrar_sesion(token: str):
    TOKENS_ACTIVOS.pop(token, None)

    return {
        "mensaje": "Sesión cerrada correctamente."
    }
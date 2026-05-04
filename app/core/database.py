import pymysql
from fastapi import HTTPException


DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "tfg_fraudeElectrico",
    "cursorclass": pymysql.cursors.DictCursor
}


# Conexión con la Base de Datos.
def obtener_conexion():
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible conectar con la base de datos."
        )


# Recuperar usuario mediante el campo del nombre.
def obtener_usuario_por_nombre(nombre_usuario: str):
    try:
        conexion = obtener_conexion()

        with conexion:
            with conexion.cursor() as cursor:
                sql = """
                    SELECT id_usuario, nombre_usuario, password_hash, rol, activo
                    FROM usuarios
                    WHERE nombre_usuario = %s
                    LIMIT 1
                """
                cursor.execute(sql, (nombre_usuario,))
                usuario = cursor.fetchone()

        return usuario

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible consultar el usuario en la base de datos."
        )
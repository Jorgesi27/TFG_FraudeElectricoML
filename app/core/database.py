import json
import pymysql
import os
from dotenv import load_dotenv
from fastapi import HTTPException
from app.core.utils import limpiar_para_json

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "cursorclass": pymysql.cursors.DictCursor
}

# Conexión con la Base de datos.
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
    
# Guarda un archivo importado por un usuario.
def guardar_archivo(id_usuario: int, nombre_archivo: str):

    try:

        conexion = obtener_conexion()

        with conexion:
            with conexion.cursor() as cursor:

                sql = """
                    INSERT INTO archivos_consumo
                    (
                        id_usuario,
                        nombre_archivo,
                        total_curvas
                    )
                    VALUES (%s, %s, %s)
                """

                cursor.execute(
                    sql,
                    (
                        id_usuario,
                        nombre_archivo,
                        0
                    )
                )

                conexion.commit()

                return cursor.lastrowid

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Guarda una curva de consumo asociada a un archivo.
def guardar_curva(
    id_archivo: int,
    identificador_curva: str,
    datos_consumo
):
    try:
        conexion = obtener_conexion()

        with conexion:
            with conexion.cursor() as cursor:

                sql = """
                    INSERT INTO curvas_consumo
                    (
                        id_archivo,
                        identificador_curva,
                        datos_consumo
                    )
                    VALUES (%s, %s, %s)
                """

                cursor.execute(
                    sql,
                    (
                        id_archivo,
                        identificador_curva,
                        json.dumps(limpiar_para_json(datos_consumo))
                    )
                )

                conexion.commit()

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible guardar una curva de consumo."
        )
    
# Recupera todas las curvas asociadas a un archivo importado.
def obtener_curvas_archivo(
    id_archivo,
    id_usuario
):
    try:
        conexion = obtener_conexion()

        with conexion:
            with conexion.cursor() as cursor:

                sql = """
                    SELECT c.*
                    FROM curvas_consumo c
                    JOIN archivos_consumo a
                        ON c.id_archivo = a.id_archivo
                    WHERE c.id_archivo = %s
                    AND a.id_usuario = %s
                """

                cursor.execute(
                    sql,
                    (
                        id_archivo,
                        id_usuario
                    )
                )

                curvas = cursor.fetchall()

        return curvas

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible recuperar "
                "las curvas del archivo."
            )
        )
    

# Recupera una curva de consumo concreta mediante su identificador.
def obtener_curva_por_id(
    id_curva: int,
    id_usuario: int
):
    try:

        conexion = obtener_conexion()

        with conexion:
            with conexion.cursor() as cursor:

                sql = """
                    SELECT
                        c.id_curva,
                        c.identificador_curva,
                        c.datos_consumo
                    FROM curvas_consumo c
                    JOIN archivos_consumo a
                        ON c.id_archivo = a.id_archivo
                    WHERE c.id_curva = %s
                    AND a.id_usuario = %s
                    LIMIT 1
                """

                cursor.execute(
                    sql,
                    (
                        id_curva,
                        id_usuario
                    )
                )

                curva = cursor.fetchone()

        if curva is None:
            return None

        datos = curva["datos_consumo"]

        # SI ES BYTES
        if isinstance(datos, bytes):
            datos = datos.decode("utf-8")

        # SI ES STRING JSON
        if isinstance(datos, str):
            datos = json.loads(datos)

        curva["datos_consumo"] = datos

        return curva

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible recuperar "
                "la curva de consumo."
            )
        )

# Recupera los archivos de un usuario por su id.
def obtener_archivos_usuario(id_usuario):

    conexion = obtener_conexion()

    with conexion:
        with conexion.cursor() as cursor:

            sql = """
                SELECT
                    id_archivo,
                    nombre_archivo
                FROM archivos_consumo
                WHERE id_usuario = %s
                ORDER BY fecha_importacion DESC
            """

            cursor.execute(sql, (id_usuario,))

            return cursor.fetchall()
        
# Guarda una predicción en la Base de datos.
def guardar_prediccion(
    id_curva,
    tipo_modelo,
    tipo_prediccion,
    resultado_prediccion,
    probabilidad_fraude
):

    conexion = obtener_conexion()

    cursor = conexion.cursor()

    query = """
        INSERT INTO predicciones (
            id_curva,
            tipo_modelo,
            tipo_prediccion,
            resultado_prediccion,
            probabilidad_fraude
        )
        VALUES (%s, %s, %s, %s, %s)
    """

    cursor.execute(
        query,
        (
            id_curva,
            tipo_modelo,
            tipo_prediccion,
            resultado_prediccion,
            probabilidad_fraude
        )
    )

    conexion.commit()

    cursor.close()

    conexion.close()

# Guarda las estadísticas de un archivo en la Base de datos.
def guardar_estadisticas_archivo(
    id_archivo: int,
    estadisticas: dict
):

    conexion = obtener_conexion()

    with conexion:
        with conexion.cursor() as cursor:

            sql = """
                UPDATE archivos_consumo
                SET estadisticas_json = %s
                WHERE id_archivo = %s
            """

            cursor.execute(
                sql,
                (
                    json.dumps(estadisticas),
                    id_archivo
                )
            )

            conexion.commit()

# Recupera estadísticas precalculadas
def obtener_estadisticas_archivo_bd(
    id_archivo: int,
    id_usuario: int
):

    conexion = obtener_conexion()

    with conexion:
        with conexion.cursor() as cursor:

            sql = """
                SELECT estadisticas_json
                FROM archivos_consumo
                WHERE id_archivo = %s
                AND id_usuario = %s
                LIMIT 1
            """

            cursor.execute(
                sql,
                (
                    id_archivo,
                    id_usuario
                )
            )

            resultado = cursor.fetchone()

    if not resultado:
        return None

    estadisticas = resultado["estadisticas_json"]

    if estadisticas is None:
        return None

    if isinstance(estadisticas, bytes):
        estadisticas = estadisticas.decode("utf-8")

    if isinstance(estadisticas, str):
        estadisticas = json.loads(estadisticas)

    estadisticas = limpiar_para_json(estadisticas)

    return estadisticas

# Eliminar un archivo del usuario, eliminando en cascada las predicciones y curvas del archivo.
def eliminar_archivo(id_archivo: int, id_usuario: int):
    try:
        conexion = obtener_conexion()
        with conexion:
            with conexion.cursor() as cursor:
                cursor.execute("""
                    DELETE p FROM predicciones p
                    JOIN curvas_consumo c ON p.id_curva = c.id_curva
                    WHERE c.id_archivo = %s
                """, (id_archivo,))

                cursor.execute("""
                    DELETE FROM curvas_consumo WHERE id_archivo = %s
                """, (id_archivo,))

                cursor.execute("""
                    DELETE FROM archivos_consumo 
                    WHERE id_archivo = %s AND id_usuario = %s
                """, (id_archivo, id_usuario))

                conexion.commit()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible eliminar el archivo."
        )
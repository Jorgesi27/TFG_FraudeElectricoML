import json
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
                        nombre_archivo
                    )
                    VALUES (%s, %s)
                """

                cursor.execute(sql, (id_usuario, nombre_archivo))
                conexion.commit()

                return cursor.lastrowid

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No ha sido posible guardar el archivo importado."
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
                        json.dumps(datos_consumo)
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
# Recupera una curva concreta
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

        print("ERROR CURVA:", e)

        raise HTTPException(
            status_code=500,
            detail=(
                "No ha sido posible recuperar "
                "la curva de consumo."
            )
        )
        

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
        

def obtener_datos_curva(id_curva):

    conexion = obtener_conexion()

    with conexion:
        with conexion.cursor() as cursor:

            sql = """
                SELECT
                    identificador_curva,
                    datos_consumo
                FROM curvas_consumo
                WHERE id_curva = %s
                LIMIT 1
            """

            cursor.execute(sql, (id_curva,))

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
# TFG - Detección de Fraude con Machine Learning

Este proyecto forma parte de mi Trabajo de Fin de Grado y tiene como objetivo desarrollar modelos de Machine Learning para la detección de fraude en el consumo eléctrico.

El proyecto incluye:
- Implementación y entrenamiento offline y online de distintos modelos de clasificación
- Evaluación mediante métricas como F1-score, precision, recall y Cohen's Kappa
- Validación cruzada para medir la estabilidad de los modelos
- Creación de una API REST donde comparar y ver resultados del mejor modelo de cada categoria

Los modelos entrenados no se incluyen en el repositorio público. Deben generarse ejecutando los scripts de entrenamiento de la carpeta experiments.

Los datasets utilizados no se incluyen en el repositorio público debido a su tamaño.

Los modelos entrenados no se incluyen en el repositorio público. Deben generarse ejecutando los scripts de entrenamiento de la carpeta experiments.

## Replicación del entorno local

Para ejecutar el sistema es necesario disponer de un entorno con soporte para Python y un sistema gestor de base de datos MySQL. Además, se requiere un navegador web moderno para acceder a la interfaz gráfica del prototipo. Los requisitos mínimos del entorno son los siguientes: Python 3.10.11, MySQL Server, Git (opcional), navegador web moderno y un sistema operativo Windows o Linux.

El proyecto puede obtenerse mediante clonación del repositorio utilizando Git o descargando directamente el código fuente en formato comprimido. La clonación del repositorio puede realizarse mediante el siguiente comando: `git clone https://github.com/Jorgesi27/TFG\_FraudeElectricoML.git`.

Una vez descargado el proyecto, debe accederse al directorio raíz del sistema: `cd TFG\_FraudeElectricoML`. Para aislar las dependencias del proyecto se utiliza un entorno virtual de Python. La creación del entorno virtual se realiza mediante el siguiente comando: `python -m venv venv`. Activación del entorno virtual en Windows: `venv\textbackslash Scripts\textbackslash activate` Activación del entorno virtual en Linux: `source venv/bin/activate`.

Una vez activado el entorno virtual, deben instalarse las bibliotecas necesarias definidas en el archivo requirements.txt. `pip install -r requirements.txt`. Para crear y configurar la base de datos utilizada por el sistema es necesario utilizar \emph{MySQL}. En primer lugar, debe crearse la base de datos principal empleada por la aplicación. `CREATE DATABASE tfg\_fraudeelectrico;`. Posteriormente, debe seleccionarse la base de datos creada. `USE tfg\_fraudeelectrico;`.

Tras la creación de la base de datos, deben configurarse las credenciales de acceso utilizadas por la aplicación. Para ello, el sistema utiliza un archivo de variables de entorno denominado .env, situado en la raíz del proyecto. Este archivo debe contener los parámetros necesarios para establecer la conexión con la base de datos MySQL, incluyendo el host, el usuario, la contraseña y el nombre de la base de datos utilizada por el sistema.

Estas variables son cargadas automáticamente por la aplicación desde el módulo encargado de la conexión con la base de datos (database.py). Además, el sistema requiere disponer de los modelos previamente entrenados utilizados durante la fase experimental. Dichos modelos se encuentran almacenados en formato .pkl dentro del directorio models/.

Por otra parte, los archivos utilizados para realizar predicciones históricas o simulaciones en tiempo real deben almacenarse dentro del directorio data/. El sistema accede a estos archivos durante la ejecución para procesar las curvas de consumo eléctrico importadas por el operador. 

Debido a que las contraseñas no se almacenan en texto plano, es necesario generar previamente el hash de la contraseña antes de insertar un usuario en la base de datos. Una vez generado el hash, el usuario puede insertarse manualmente en la tabla usuarios, usando el usuario y contraseña con hash del script hash.py, mediante una sentencia SQL como la del script insertar_usuario.sql.

Finalmente, una vez completada toda la configuración, la aplicación puede iniciarse eejcutando el servidor \emph{Uvicorn} utilizando el siguiente comando: \texttt{uvicorn app.main:app --reload}

Tras la ejecución del servidor, la interfaz web y la documentación automática de la API estarán disponibles desde el navegador web. Por defecto, el servidor se ejecuta sobre el puerto 8000. Por tanto, la aplicación web puede accederse desde la dirección: http://localhost:8000. Y la documentación interactiva generada automáticamente por FastAPI se encuentra disponible en: http://localhost:8000/docs.

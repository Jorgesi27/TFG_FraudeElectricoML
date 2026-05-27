import {
    fetchAuth,
    obtenerData
} from "./api.js";

import {
    limpiarGrafica,
    crearGraficaLineal
} from "./charts.js";

import {
    mostrarResultado
} from "./predicciones.js";

import {
    mostrarError
} from "./ui.js";

import {
    logout,
    redirigirLogin
} from "./auth.js";

// Elementos DOM
const archivoSelect = document.getElementById("archivoSelect");
const curvaSelect = document.getElementById("curvaSelect");
const resultadoTiempoReal = document.getElementById("resultadoTiempoReal");
const probabilidadTiempoReal = document.getElementById("probabilidadTiempoReal");
const graficaTiempoReal = document.getElementById("graficaTiempoReal");

// Variables
let chartTiempoReal = null;
let intervaloStream = null;
let puntosConsumo = [];
let labelsTiempo = [];
let prediccionEnCurso = false;

// Limpia los resultados mostrados en pantalla.
function limpiarResultados(){

    resultadoTiempoReal.innerText =
        "---";

    probabilidadTiempoReal.innerText =
        "---";

    resultadoTiempoReal.style.color =
        "#ffffff";
}

// Recupera los archivos importados por el usuario.
async function cargarArchivos(){

    try{

        const response = await fetchAuth(
            "/api/operador/archivos"
        );

        if(!response){
            return;
        }

        const archivos =
            await obtenerData(response);

        archivoSelect.innerHTML =

            `
            <option value="">
                Seleccionar archivo
            </option>
            `;

        archivos.forEach(archivo => {

            archivoSelect.innerHTML +=

                `
                <option value="${archivo.id_archivo}">
                    ${archivo.nombre_archivo}
                </option>
                `;
        });

    }catch(error){

        mostrarError(
            "Error cargando archivos",
            error
        );
    }
}

// Carga las curvas asociadas al archivo seleccionado.
async function cargarCurvas(){

    const idArchivo =
        archivoSelect.value;

    curvaSelect.innerHTML =

        `
        <option value="">
            Seleccionar curva
        </option>
        `;

    if(!idArchivo){
        return;
    }

    try{

        const response = await fetchAuth(

            `/api/operador/curvas/${idArchivo}`
        );

        if(!response){
            return;
        }

        const data =
            await obtenerData(response);

        data.curvas.forEach(curva => {

            curvaSelect.innerHTML +=

                `
                <option value="${curva.id_curva}">
                    ${curva.identificador_curva}
                </option>
                `;
        });

    }catch(error){

        mostrarError(
            "Error cargando curvas",
            error
        );
    }
}

// Simula el flujo de consumo en tiempo real y realiza predicciones online.
async function iniciarStreaming(){

    const idCurva =
        curvaSelect.value;

    if(!idCurva){
        return;
    }

    // LIMPIAR ANTERIOR

    if(intervaloStream){

        clearInterval(
            intervaloStream
        );
    }

    try{
        // OBTENER CURVA

        const response = await fetchAuth(

            `/api/operador/curva/${idCurva}`
        );

        if(!response){
            return;
        }

        const curva =
            await obtenerData(response);

        const labelsOriginales =
            curva.labels || [];

        const valores =
            (curva.valores || [])
                .map(v => Number(v) || 0)

        if(!valores.length){

            mostrarError(
                "La curva no contiene valores"
            );

            return;
        }

        // REINICIAR GRAFICA
 
        puntosConsumo = [];

        labelsTiempo = [];

        prediccionEnCurso = false;

        chartTiempoReal =
            limpiarGrafica(
                chartTiempoReal
            );

        chartTiempoReal =
            crearGraficaLineal(

                graficaTiempoReal,

                labelsTiempo,

                puntosConsumo,

                curva.identificador_curva
            );

        // FLUJO ONLINE

        let indice = 0;

        intervaloStream = setInterval(async () => {

            // FIN STREAM
            if(indice >= valores.length){

                clearInterval(intervaloStream);

                intervaloStream = null;

                return;
            }

            try{

                // NUEVO PUNTO
                puntosConsumo.push(
                    Number(valores[indice])
                );

                const minutos = String(
                    Math.floor(indice / 60)
                ).padStart(2, "0");

                const segundos = String(
                    indice % 60
                ).padStart(2, "0");

                labelsTiempo.push(
                    `${minutos}:${segundos}`
                );

                // ACTUALIZAR
                chartTiempoReal.data.labels = labelsTiempo;

                chartTiempoReal.data.datasets[0].data =
                    puntosConsumo;

                chartTiempoReal.update();

                const PASO_VENTANA = 48;

                // SOLO PREDECIR EN MULTIPLOS DE 48
                if((indice + 1) % PASO_VENTANA !== 0){

                    indice++;

                    return;
                }

                if(prediccionEnCurso){

                    indice++;

                    return;
                }

                prediccionEnCurso = true;

                const tamañoVentana = indice + 1;

                const datosParciales =
                    valores
                        .slice(0, tamañoVentana)
                        .filter(
                            v =>
                                v !== undefined &&
                                v !== null &&
                                !isNaN(v)
                        );

                if(!datosParciales.length){
                    prediccionEnCurso = false;
                    indice++;
                    return;
                }

                resultadoTiempoReal.innerHTML =
                    `
                    <div class="badge-analizando fade-in">
                        🟡 Analizando ventana ${tamañoVentana}
                    </div>
                    `;

                const predResponse =
                    await fetchAuth(
                        "/api/operador/prediccion/stream",
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body: JSON.stringify({
                                valores: datosParciales
                            })
                        }
                    );

                if(predResponse && predResponse.ok){

                    const prediccion =
                        await obtenerData(predResponse);

                    mostrarResultado(
                        resultadoTiempoReal,
                        probabilidadTiempoReal,
                        prediccion
                    );

                }

                prediccionEnCurso = false;

                // AUMENTAR INDICE
                indice++;

            }catch(error){

                prediccionEnCurso = false;

                clearInterval(intervaloStream);

                intervaloStream = null;

                mostrarError(
                    "Error en streaming",
                    error
                );
            }

        }, 3000);

    }catch(error){

        mostrarError(

            "Error streaming tiempo real",

            error
        );
    }
}

// Inicializa los eventos principales de la interfaz.
function inicializarEventos(){

    document
        .getElementById("btnLogout")
        .addEventListener(
            "click",
            logout
        );

    archivoSelect.addEventListener(
        "change",
        cargarCurvas
    );

    curvaSelect.addEventListener(
        "change",
        iniciarStreaming
    );
}

// Inicializa la pantalla al cargar la aplicación.
window.onload = async () => {

    const token =
        localStorage.getItem("token");

    if(!token){

        redirigirLogin();

        return;
    }

    inicializarEventos();

    limpiarResultados();

    await cargarArchivos();
};
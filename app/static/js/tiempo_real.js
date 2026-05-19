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


// ========================================
// ELEMENTOS DOM
// ========================================

const archivoSelect =
    document.getElementById("archivoSelect");

const curvaSelect =
    document.getElementById("curvaSelect");

const resultadoTiempoReal =
    document.getElementById(
        "resultadoTiempoReal"
    );

const probabilidadTiempoReal =
    document.getElementById(
        "probabilidadTiempoReal"
    );

const graficaTiempoReal =
    document.getElementById(
        "graficaTiempoReal"
    );


// ========================================
// VARIABLES
// ========================================

let chartTiempoReal = null;

let intervaloStream = null;

let puntosConsumo = [];

let labelsTiempo = [];


// ========================================
// LIMPIAR
// ========================================

function limpiarResultados(){

    resultadoTiempoReal.innerText =
        "---";

    probabilidadTiempoReal.innerText =
        "---";

    resultadoTiempoReal.style.color =
        "#ffffff";
}


// ========================================
// CARGAR ARCHIVOS
// ========================================

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


// ========================================
// CARGAR CURVAS
// ========================================

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


// ========================================
// STREAMING TIEMPO REAL
// ========================================

async function iniciarStreaming(){

    const idCurva =
        curvaSelect.value;

    if(!idCurva){
        return;
    }

    // =========================
    // LIMPIAR STREAM ANTERIOR
    // =========================

    if(intervaloStream){

        clearInterval(
            intervaloStream
        );
    }

    try{

        // =========================
        // OBTENER CURVA
        // =========================

        const response = await fetchAuth(

            `/api/operador/curva/${idCurva}`
        );

        if(!response){
            return;
        }

        const curva =
            await obtenerData(response);

        // =========================
        // EXTRAER VALORES
        // =========================

        const valores = Object.entries(
            curva.datos_consumo
        )
        .filter(([_, valor]) =>
            typeof valor === "number"
        )
        .map(([_, valor]) => valor);

        // =========================
        // RESETEAR GRAFICA
        // =========================

        puntosConsumo = [];

        labelsTiempo = [];

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

        // =========================
        // STREAMING
        // =========================

        let indice = 0;

        intervaloStream = setInterval(

            async () => {

                // =========================
                // FIN STREAM
                // =========================

                if(indice >= valores.length){

                    clearInterval(
                        intervaloStream
                    );

                    return;
                }

                // =========================
                // NUEVO PUNTO
                // =========================

                puntosConsumo.push(
                    valores[indice]
                );

                const fecha = new Date(

                    2024,
                    0,
                    1,
                    0,
                    indice
                );

                labelsTiempo.push(

                    fecha.toLocaleTimeString(
                        "es-ES",
                        {
                            hour: "2-digit",
                            minute: "2-digit"
                        }
                    )
                );

                // =========================
                // ACTUALIZAR GRAFICA
                // =========================

                chartTiempoReal.update();

                // =========================
                // DATOS PARCIALES
                // =========================

                const datosParciales =

                    valores.slice(
                        0,
                        indice + 1
                    );

                // =========================
                // LLAMAR AL MODELO REAL
                // =========================

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

                                valores:
                                    datosParciales
                            })
                        }
                    );

                // =========================
                // ACTUALIZAR RESULTADO
                // =========================

                if(predResponse){

                    const prediccion =
                        await obtenerData(
                            predResponse
                        );

                    mostrarResultado(

                        resultadoTiempoReal,

                        probabilidadTiempoReal,

                        prediccion
                    );
                }

                indice++;

            },

            3000
        );

    }catch(error){

        mostrarError(

            "Error streaming tiempo real",

            error
        );
    }
}


// ========================================
// EVENTOS
// ========================================

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


// ========================================
// INIT
// ========================================

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
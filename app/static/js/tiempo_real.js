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

// =====================================================
// VARIABLES
// =====================================================

let archivoSelect = null;
let curvaSelect = null;
let resultadoTiempoReal = null;
let probabilidadTiempoReal = null;
let graficaTiempoReal = null;

let chartTiempoReal = null;
let intervaloStream = null;
let puntosConsumo = [];
let labelsTiempo = [];

// =====================================================
// LIMPIAR RESULTADOS
// =====================================================

function limpiarResultados(){
    resultadoTiempoReal.innerHTML = "---";
    probabilidadTiempoReal.innerHTML = "---";
}

// =====================================================
// CARGAR ARCHIVOS
// =====================================================

async function cargarArchivos(){

    try{
        const response = await fetchAuth("/api/operador/archivos");
        if(!response){ return; }

        const archivos = await obtenerData(response);

        archivoSelect.innerHTML = `<option value="">Seleccionar archivo</option>`;

        archivos.forEach(archivo => {
            archivoSelect.innerHTML += `
                <option value="${archivo.id_archivo}">
                    ${archivo.nombre_archivo}
                </option>
            `;
        });

    }catch(error){
        mostrarError("Error cargando archivos", error);
    }
}

// =====================================================
// CARGAR CURVAS
// =====================================================

async function cargarCurvas(){

    const idArchivo = archivoSelect.value;

    curvaSelect.innerHTML = `<option value="">Seleccionar curva</option>`;

    if(!idArchivo){ return; }

    try{
        const response = await fetchAuth(`/api/operador/curvas/${idArchivo}`);
        if(!response){ return; }

        const data = await obtenerData(response);

        data.curvas.forEach(curva => {
            curvaSelect.innerHTML += `
                <option value="${curva.id_curva}">
                    ${curva.identificador_curva}
                </option>
            `;
        });

    }catch(error){
        mostrarError("Error cargando curvas", error);
    }
}

// =====================================================
// STREAMING ONLINE
// =====================================================
async function iniciarStreaming(){

    const idCurva = curvaSelect.value;
    if(!idCurva){ return; }

    if(intervaloStream){
        clearInterval(intervaloStream);
        intervaloStream = null;
    }

    puntosConsumo = [];
    labelsTiempo = [];
    limpiarResultados();
    chartTiempoReal = limpiarGrafica(chartTiempoReal);

    resultadoTiempoReal.innerHTML =
        `<div class="badge-analizando">⏳ Cargando predicciones...</div>`;

    try{

        const response = await fetchAuth(
            `/api/operador/prediccion/tiempo-real/${idCurva}`,
            { method: "POST" }
        );

        if(!response){ return; }

        const data = await obtenerData(response);
        const predicciones = data.predicciones || [];
        const tramoInicial = data.tramo_inicial || [];

        if(!predicciones.length){
            mostrarError("Sin predicciones disponibles");
            return;
        }

        // Rango fijo del eje Y considerando TODO el consumo (tramo inicial + predicciones)
        const consumosCurva = [
            ...tramoInicial.map(p => p.consumo),
            ...predicciones.map(p => p.consumo)
        ];
        const minConsumo = Math.min(...consumosCurva);
        const maxConsumo = Math.max(...consumosCurva);
        const margen = (maxConsumo - minConsumo) * 0.1;

        const yMin = Math.max(0, minConsumo - margen);
        const yMax = maxConsumo + margen;

        // Pintar de golpe el tramo inicial (sin clasificar)
        puntosConsumo = tramoInicial.map(p => p.consumo);
        labelsTiempo = tramoInicial.map(p => p.hora);

        chartTiempoReal = crearGraficaLineal(
            graficaTiempoReal,
            labelsTiempo,
            puntosConsumo,
            data.identificador_curva,
            yMin,
            yMax
        );

        limpiarResultados();

        let indice = 0;

        intervaloStream = setInterval(() => {

            if(indice >= predicciones.length){
                clearInterval(intervaloStream);
                intervaloStream = null;
                return;
            }

            const punto = predicciones[indice];

            puntosConsumo.push(punto.consumo);
            labelsTiempo.push(punto.hora);
            chartTiempoReal.data.labels = labelsTiempo;
            chartTiempoReal.data.datasets[0].data = puntosConsumo;
            chartTiempoReal.update("none");

            mostrarResultado(
                resultadoTiempoReal,
                probabilidadTiempoReal,
                {
                    estado: "online",
                    resultado: punto.fraude === 1 ? "Fraude" : "Normal",
                    probabilidad: punto.probabilidad
                }
            );

            indice++;

        }, 3000);

    }catch(error){
        mostrarError("Error iniciando streaming", error);
    }
}

// =====================================================
// EVENTOS
// =====================================================

function inicializarEventos(){

    document
        .getElementById("btnLogout")
        .addEventListener("click", logout);

    archivoSelect.addEventListener("change", cargarCurvas);

    curvaSelect.addEventListener("change", iniciarStreaming);
}

// =====================================================
// INIT
// =====================================================

window.onload = async () => {

    const token = localStorage.getItem("token");

    if(!token){
        redirigirLogin();
        return;
    }

    archivoSelect = document.getElementById("archivoSelect");
    curvaSelect = document.getElementById("curvaSelect");
    resultadoTiempoReal = document.getElementById("resultadoTiempoReal");
    probabilidadTiempoReal = document.getElementById("probabilidadTiempoReal");
    graficaTiempoReal = document.getElementById("graficaTiempoReal");

    inicializarEventos();
    limpiarResultados();
    await cargarArchivos();
};
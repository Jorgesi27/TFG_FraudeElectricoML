/*
import {
    fetchAuth,
    obtenerData
} from "./api.js";

import {
    mostrarElemento,
    ocultarElemento,
    mostrarError,
    mostrarFlex
} from "./ui.js";

import {
    limpiarGrafica,
    crearGraficaLineal,
    crearGraficaPie,
    crearGraficaBarras
} from "./charts.js";

import {
    prediccionHistorica,
    mostrarResultado
} from "./predicciones.js";

import {
    obtenerEstadisticas
} from "./estadisticas.js";

import {
    redirigirLogin,
    logout
} from "./auth.js";


// ========================================
// ELEMENTOS DOM
// ========================================

const archivoSelect =
    document.getElementById("archivoSelect");

const curvaSelect =
    document.getElementById("curvaSelect");

const resultadoHistorico =
    document.getElementById("resultadoHistorico");

const probabilidadHistorica =
    document.getElementById("probabilidadHistorica");

const graficaHistorica =
    document.getElementById("graficaHistorica");

const graficaEstadisticas =
    document.getElementById("graficaEstadisticas");

const graficaBarras =
    document.getElementById("graficaBarras");

const contenedorGraficas =
    document.getElementById("contenedorGraficas");


// ========================================
// GRAFICAS
// ========================================

let chart = null;

let chartEstadisticas = null;

let chartBarras = null;


// ========================================
// IMPORTAR CSV
// ========================================

async function importarArchivo(){

    const input =
        document.getElementById("csvFile");

    if(!input.files.length){

        alert("Seleccione un archivo CSV");

        return;
    }

    const formData = new FormData();

    formData.append(
        "archivo",
        input.files[0]
    );

    try{

        const response = await fetchAuth(
            "/api/operador/importar-csv",
            {
                method:"POST",
                body:formData
            }
        );

        if(!response){
            return;
        }

        await obtenerData(response);

        alert("Archivo importado correctamente");

        input.value = "";

        await cargarArchivos();

    }catch(error){

        mostrarError(
            "Error importando archivo",
            error
        );
    }
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
            `<option value="">
                Seleccionar archivo
            </option>`;

        archivos.forEach(archivo => {

            archivoSelect.innerHTML += `
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
        `<option value="">
            Seleccionar curva
        </option>`;

    limpiarVista();

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

            curvaSelect.innerHTML += `
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
// LIMPIAR VISTA
// ========================================

function limpiarResultados(){

    resultadoHistorico.innerText = "---";

    probabilidadHistorica.innerText = "---";

    resultadoHistorico.style.color =
        "#ffffff";
}


function limpiarVista(){

    limpiarResultados();

    chart =
        limpiarGrafica(chart);

    chartEstadisticas =
        limpiarGrafica(chartEstadisticas);

    chartBarras =
        limpiarGrafica(chartBarras);
}


// ========================================
// MOSTRAR SECCIONES
// ========================================

function ocultarTodasLasSecciones(){

    ocultarElemento(
        document.getElementById(
            "seccionHistorica"
        )
    );

    ocultarElemento(
        document.getElementById(
            "seccionTiempoReal"
        )
    );

    ocultarElemento(
        document.getElementById(
            "seccionEstadisticas"
        )
    );
}


function mostrarSeccion(seccion){

    ocultarTodasLasSecciones();

    if(seccion === "historica"){

        mostrarElemento(
            document.getElementById(
                "seccionHistorica"
            )
        );
    }

    if(seccion === "tiemporeal"){

        mostrarElemento(
            document.getElementById(
                "seccionTiempoReal"
            )
        );
    }

    if(seccion === "estadisticas"){

        mostrarElemento(
            document.getElementById(
                "seccionEstadisticas"
            )
        );
    }
}


// ========================================
// MOSTRAR GRAFICA
// ========================================

async function mostrarGrafica(){

    const idCurva =
        curvaSelect.value;

    limpiarResultados();

    if(!idCurva){

        chart =
            limpiarGrafica(chart);

        return;
    }

    try{

        const response = await fetchAuth(
            `/api/operador/curva/${idCurva}`
        );

        if(!response){
            return;
        }

        const curva =
            await obtenerData(response);

        const valores =
            Object.values(
                curva.datos_consumo
            ).filter(
                valor =>
                    typeof valor === "number"
            );

        const labels =
            valores.map((_, i) => {

                const hora =
                    String(i)
                        .padStart(2, "0");

                return `${hora}:00`;
            });

        chart =
            limpiarGrafica(chart);

        chart = crearGraficaLineal(
            graficaHistorica,
            labels,
            valores,
            curva.identificador_curva
        );

    }catch(error){

        mostrarError(
            "Error mostrando gráfica",
            error
        );
    }
}


async function ejecutarPrediccionHistorica(){

    const idCurva = curvaSelect.value;

    if(!idCurva){
        alert("Seleccione una curva");
        return;
    }

    try{

        const prediccion =
            await prediccionHistorica(idCurva);

        if(!prediccion){
            return;
        }

        mostrarResultado(
            resultadoHistorico,
            probabilidadHistorica,
            prediccion
        );

    }catch(error){

        mostrarError(
            "Error realizando predicción",
            error
        );
    }
}


// ========================================
// ESTADISTICAS
// ========================================

async function cargarEstadisticas(){

    const idArchivo =
        archivoSelect.value;

    if(!idArchivo){

        alert("Seleccione un archivo");

        return;
    }

    try{

        const data =
            await obtenerEstadisticas(
                idArchivo
            );

        chart =
            limpiarGrafica(chart);

        mostrarFlex(
            contenedorGraficas
        );

        chartEstadisticas =
            limpiarGrafica(
                chartEstadisticas
            );

        chartBarras =
            limpiarGrafica(
                chartBarras
            );

        chartEstadisticas =
            crearGraficaPie(
                graficaEstadisticas,
                data
            );

        chartBarras =
            crearGraficaBarras(
                graficaBarras,
                data
            );

    }catch(error){

        mostrarError(
            "Error cargando estadísticas",
            error
        );
    }
}


// ========================================
// EVENTOS
// ========================================

function inicializarEventos(){

    document
        .getElementById("btnPredecirHistorica")
        .addEventListener(
            "click",
            ejecutarPrediccionHistorica
        );

    document
        .getElementById("btnTiempoReal")
        .addEventListener(
            "click",
            () => mostrarSeccion("tiemporeal")
        );

    document
        .getElementById("btnEstadisticas")
        .addEventListener(
            "click",
            async () => {

                mostrarSeccion(
                    "estadisticas"
                );

                await cargarEstadisticas();
            }
        );

    document
        .getElementById("btnLogout")
        .addEventListener(
            "click",
            logout
        );

    document
        .getElementById("btnImportar")
        .addEventListener(
            "click",
            importarArchivo
        );

    archivoSelect.addEventListener(
        "change",
        cargarCurvas
    );

    curvaSelect.addEventListener(
        "change",
        mostrarGrafica
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

    await cargarArchivos();

    mostrarSeccion("historica");
};
*/
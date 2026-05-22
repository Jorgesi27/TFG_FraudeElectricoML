import {
    fetchAuth,
    obtenerData
} from "./api.js";

import {
    activarMenu
} from "./menu.js";

import {
    limpiarGrafica,
    crearGraficaLineal
} from "./charts.js";

import {
    prediccionHistorica,
    mostrarResultado
} from "./predicciones.js";

import {
    mostrarError
} from "./ui.js";

import {
    logout,
    redirigirLogin
} from "./auth.js";


let archivoSelect = null;
let curvaSelect = null;
let resultadoHistorico = null;
let probabilidadHistorica = null;
let graficaHistorica = null;


// ========================================
// GRAFICA
// ========================================

let chart = null;

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

        console.error(error);

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

    limpiarResultados();

    chart =
        limpiarGrafica(chart);

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
// LIMPIAR RESULTADOS
// ========================================

function limpiarResultados(){

    resultadoHistorico.innerText =
        "---";

    probabilidadHistorica.innerText =
        "---";

    resultadoHistorico.style.color =
        "#ffffff";
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

        const labels =
            curva.datos_consumo.timestamps;

        const valores =
            curva.datos_consumo.values;

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


// ========================================
// REALIZAR PREDICCION
// ========================================

async function ejecutarPrediccionHistorica(){

    const idCurva =
        curvaSelect.value;

    if(!idCurva){

        alert("Seleccione una curva");

        return;
    }

    try{

        const prediccion =

            await prediccionHistorica(
                idCurva
            );

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
// EVENTOS
// ========================================

function inicializarEventos(){

    const btnLogout =
        document.getElementById("btnLogout");

    if(btnLogout){

        btnLogout.addEventListener(
            "click",
            logout
        );
    }else{

        console.error(
            "No existe #btnLogout"
        );
    }

    if(archivoSelect){

        archivoSelect.addEventListener(
            "change",
            cargarCurvas
        );
    }else{

        console.error(
            "No existe #archivoSelect"
        );
    }

    if(curvaSelect){

        curvaSelect.addEventListener(
            "change",
            async () => {

                await mostrarGrafica();

                await ejecutarPrediccionHistorica();
            }
        );

    }else{

        console.error(
            "No existe #curvaSelect"
        );
    }
}


// ========================================
// INIT
// ========================================

document.addEventListener(
    "DOMContentLoaded",
    async () => {

        const token =
            localStorage.getItem("token");

        if(!token){

            redirigirLogin();

            return;
        }

        archivoSelect =
            document.getElementById("archivoSelect");

        curvaSelect =
            document.getElementById("curvaSelect");

        resultadoHistorico =
            document.getElementById("resultadoHistorico");

        probabilidadHistorica =
            document.getElementById("probabilidadHistorica");

        graficaHistorica =
            document.getElementById("graficaHistorica");

        inicializarEventos();

        activarMenu();

        await cargarArchivos();
    }
);
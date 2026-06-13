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
let chart = null;
let porcentajeHorasFraude = null;

// Carga los archivos disponibles del usuario.
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

// Limpia los resultados mostrados en pantalla.
function limpiarResultados(){

    resultadoHistorico.innerText =
        "---";

    probabilidadHistorica.innerHTML  =
        "---";

    resultadoHistorico.style.color =
        "#ffffff";
}

// Muestra la gráfica de la curva seleccionada.
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
            curva.labels || [];

        const valores =
            curva.valores || [];

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

// Ejecuta la predicción histórica de la curva.
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

        if(porcentajeHorasFraude){
            porcentajeHorasFraude.innerText =
                `Horas en fraude: ${prediccion.porcentaje_horas_fraude}`;
        }

    }catch(error){

        mostrarError(
            "Error realizando predicción",
            error
        );
    }
}

// Inicializa todos los eventos de la interfaz.
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

// Inicializa la pantalla al cargar el documento.
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

        porcentajeHorasFraude =
            document.getElementById("porcentajeHorasFraude");

        inicializarEventos();

        activarMenu();

        await cargarArchivos();
    }
);
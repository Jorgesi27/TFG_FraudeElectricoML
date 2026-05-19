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


// ========================================
// GRAFICA
// ========================================

let chart = null;


// ========================================
// IMPORTAR CSV
// ========================================

async function importarArchivo(){

    const loader =
        document.getElementById(
            "loaderImportar"
        );

    loader.style.display = "block";

    const input =
        document.getElementById("csvFile");

    if(!input.files.length){

        loader.style.display = "none";

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

        alert(
            "Archivo importado correctamente"
        );

        input.value = "";

        await cargarArchivos();

    }catch(error){

        mostrarError(
            "Error importando archivo",
            error
        );

    }finally{

        loader.style.display = "none";
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

        const valores = Object.entries(
            curva.datos_consumo
        )
        .filter(([_, valor]) =>
            typeof valor === "number"
        )
        .map(([_, valor]) => valor);

        const labels = valores.map((_, i) => {

            const fecha = new Date(
                2024,
                0,
                1,
                0,
                i
            );

            return fecha.toLocaleTimeString(
                "es-ES",
                {
                    hour: "2-digit",
                    minute: "2-digit"
                }
            );
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

    document
        .getElementById("btnImportar")
        .addEventListener(
            "click",
            importarArchivo
        );

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
        async () => {

            await mostrarGrafica();

            await ejecutarPrediccionHistorica();
        }
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
    activarMenu();
    await cargarArchivos();
};
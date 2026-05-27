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
// ELEMENTOS DOM
// =====================================================

const archivoSelect =
    document.getElementById("archivoSelect");

const curvaSelect =
    document.getElementById("curvaSelect");

const resultadoTiempoReal =
    document.getElementById("resultadoTiempoReal");

const probabilidadTiempoReal =
    document.getElementById("probabilidadTiempoReal");

const graficaTiempoReal =
    document.getElementById("graficaTiempoReal");

// =====================================================
// VARIABLES
// =====================================================

let chartTiempoReal = null;

let intervaloStream = null;

let puntosConsumo = [];

let labelsTiempo = [];

let prediccionEnCurso = false;

// =====================================================
// LIMPIAR RESULTADOS
// =====================================================

function limpiarResultados(){

    resultadoTiempoReal.innerText =
        "---";

    probabilidadTiempoReal.innerText =
        "---";

    resultadoTiempoReal.style.color =
        "#ffffff";
}

// =====================================================
// CARGAR ARCHIVOS
// =====================================================

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

// =====================================================
// CARGAR CURVAS
// =====================================================

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

// =====================================================
// STREAMING ONLINE
// =====================================================

async function iniciarStreaming(){

    const idCurva =
        curvaSelect.value;

    if(!idCurva){
        return;
    }

    // ==========================================
    // LIMPIAR STREAM ANTERIOR
    // ==========================================

    if(intervaloStream){

        clearInterval(
            intervaloStream
        );
    }

    try{

        // ==========================================
        // OBTENER CURVA
        // ==========================================

        const response = await fetchAuth(

            `/api/operador/curva/${idCurva}`
        );

        if(!response){
            return;
        }

        const curva =
            await obtenerData(response);

        const valores =
            (curva.valores || [])
                .map(v => Number(v) || 0);

        if(!valores.length){

            mostrarError(
                "La curva no contiene valores"
            );

            return;
        }

        // ==========================================
        // REINICIAR VARIABLES
        // ==========================================

        puntosConsumo = [];

        labelsTiempo = [];

        prediccionEnCurso = false;

        limpiarResultados();

        // ==========================================
        // LIMPIAR GRAFICA
        // ==========================================

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

        // ==========================================
        // STREAM ONLINE
        // ==========================================

        let indice = 0;

        intervaloStream = setInterval(async () => {

            // ==========================================
            // FIN STREAM
            // ==========================================

            if(indice >= valores.length){

                clearInterval(intervaloStream);

                intervaloStream = null;

                return;
            }

            // ==========================================
            // EVITAR SOLAPAMIENTO REQUESTS
            // ==========================================

            if(prediccionEnCurso){
                return;
            }

            prediccionEnCurso = true;

            try{

                // ==========================================
                // NUEVO PUNTO
                // ==========================================

                const valorActual =
                    Number(valores[indice]) || 0;

                puntosConsumo.push(
                    valorActual
                );

                // ==========================================
                // LABEL TIEMPO
                // ==========================================

                const minutos = String(
                    Math.floor(indice / 60)
                ).padStart(2, "0");

                const segundos = String(
                    indice % 60
                ).padStart(2, "0");

                labelsTiempo.push(
                    `${minutos}:${segundos}`
                );

                // ==========================================
                // ACTUALIZAR GRAFICA
                // ==========================================

                chartTiempoReal.data.labels =
                    labelsTiempo;

                chartTiempoReal.data.datasets[0].data =
                    puntosConsumo;

                chartTiempoReal.update();

                // ==========================================
                // PREDECIR CADA 5 PUNTOS
                // ==========================================

                const PASO_PREDICCION = 5;

                if((indice + 1) % PASO_PREDICCION !== 0){

                    indice++;

                    prediccionEnCurso = false;

                    return;
                }

                // ==========================================
                // DATOS PARCIALES
                // ==========================================

                const datosParciales =

                    valores
                        .slice(0, indice + 1)
                        .map(v => Number(v) || 0);

                // ==========================================
                // MENSAJE
                // ==========================================

                resultadoTiempoReal.innerHTML =

                    `
                    <div class="badge-analizando fade-in">
                        Analizando punto ${indice + 1}
                    </div>
                    `;

                // ==========================================
                // REQUEST API
                // ==========================================

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

                // ==========================================
                // RESPUESTA
                // ==========================================

                if(predResponse && predResponse.ok){

                    const prediccion =
                        await obtenerData(predResponse);

                    mostrarResultado(

                        resultadoTiempoReal,

                        probabilidadTiempoReal,

                        prediccion
                    );
                }

                indice++;

                prediccionEnCurso = false;

            }catch(error){

                prediccionEnCurso = false;

                clearInterval(intervaloStream);

                intervaloStream = null;

                mostrarError(
                    "Error en streaming online",
                    error
                );
            }

        }, 500);

    }catch(error){

        mostrarError(

            "Error streaming tiempo real",

            error
        );
    }
}

// =====================================================
// EVENTOS
// =====================================================

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

// =====================================================
// INIT
// =====================================================

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
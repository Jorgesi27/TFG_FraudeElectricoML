import {
    fetchAuth,
    obtenerData
} from "./api.js";

import {
    logout,
    redirigirLogin
} from "./auth.js";

import {
    activarMenu
} from "./menu.js";

import {
    mostrarError
} from "./ui.js";


// ========================================
// ELEMENTOS DOM
// ========================================

const archivoSelect =
    document.getElementById(
        "archivoSelect"
    );


// ========================================
// CHARTS
// ========================================

let pieChart = null;

let barChart = null;

let lineChart = null;

let radarChart = null;


// ========================================
// LIMPIAR CHART
// ========================================

function limpiarChart(chart){

    if(chart){

        chart.destroy();
    }

    return null;
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
// CARGAR ESTADISTICAS
// ========================================

async function cargarDashboard(){

    const idArchivo =
        archivoSelect.value;

    if(!idArchivo){

        alert(
            "Seleccione un archivo"
        );

        return;
    }

    try{

        const response = await fetchAuth(

            `/api/operador/estadisticas/${idArchivo}`
        );

        if(!response){
            return;
        }

        const data =
            await obtenerData(response);

        cargarKPIs(data);

        crearPieChart(data);

        crearBarChart(data);

        crearLineChart(data);

        crearRadarChart(data);

    }catch(error){

        mostrarError(
            "Error generando dashboard",
            error
        );
    }
}


// ========================================
// KPIs
// ========================================

function cargarKPIs(data){

    document.getElementById(
        "kpiTotal"
    ).innerText =
        data.total_curvas;

    document.getElementById(
        "kpiFraudes"
    ).innerText =
        data.fraudes;

    document.getElementById(
        "kpiNormales"
    ).innerText =
        data.normales;

    document.getElementById(
        "kpiPorcentaje"
    ).innerText =
        `${data.porcentaje_fraudes}%`;
}


// ========================================
// PIE CHART
// ========================================

function crearPieChart(data){

    const ctx =
        document
            .getElementById("graficaPie");

    pieChart =
        limpiarChart(pieChart);

    pieChart = new Chart(ctx, {

        type:"doughnut",

        data:{

            labels:[
                "Fraudes",
                "Normales"
            ],

            datasets:[{

                data:[
                    data.fraudes,
                    data.normales
                ],

                backgroundColor:[
                    "#ef4444",
                    "#3b82f6"
                ]
            }]
        },

        options:{

            responsive:true,

            plugins:{

                legend:{
                    labels:{
                        color:"white"
                    }
                }
            }
        }
    });
}


// ========================================
// BARRAS
// ========================================

function crearBarChart(data){

    const ctx =
        document
            .getElementById(
                "graficaBarras"
            );

    barChart =
        limpiarChart(barChart);

    barChart = new Chart(ctx, {

        type:"bar",

        data:{

            labels:[
                "Fraudes",
                "Normales"
            ],

            datasets:[{

                label:"Cantidad",

                data:[
                    data.fraudes,
                    data.normales
                ],

                backgroundColor:[
                    "#ef4444",
                    "#3b82f6"
                ]
            }]
        },

        options:{

            responsive:true,

            scales:{

                y:{
                    ticks:{
                        color:"white"
                    }
                },

                x:{
                    ticks:{
                        color:"white"
                    }
                }
            },

            plugins:{

                legend:{
                    labels:{
                        color:"white"
                    }
                }
            }
        }
    });
}


// ========================================
// LINEAL
// ========================================

function crearLineChart(data){

    const ctx =
        document
            .getElementById(
                "graficaLineal"
            );

    lineChart =
        limpiarChart(lineChart);

    const puntos = [];

    for(let i = 0; i < data.total_curvas; i++){

        puntos.push(

            Math.floor(
                Math.random() * 100
            )
        );
    }

    lineChart = new Chart(ctx, {

        type:"line",

        data:{

            labels:
                puntos.map(
                    (_, i) => `T${i+1}`
                ),

            datasets:[{

                label:
                    "Riesgo temporal",

                data:puntos,

                borderColor:"#3b82f6",

                backgroundColor:
                    "rgba(59,130,246,0.2)",

                tension:0.4,

                fill:true
            }]
        },

        options:{

            responsive:true,

            plugins:{

                legend:{
                    labels:{
                        color:"white"
                    }
                }
            },

            scales:{

                y:{
                    ticks:{
                        color:"white"
                    }
                },

                x:{
                    ticks:{
                        color:"white"
                    }
                }
            }
        }
    });
}


// ========================================
// RADAR
// ========================================

function crearRadarChart(data){

    const ctx =
        document
            .getElementById(
                "graficaRadar"
            );

    radarChart =
        limpiarChart(radarChart);

    radarChart = new Chart(ctx, {

        type:"radar",

        data:{

            labels:[

                "Fraudes",
                "Normales",
                "Riesgo",
                "Consumo",
                "Detección"
            ],

            datasets:[{

                label:"Indicadores",

                data:[

                    data.fraudes,

                    data.normales,

                    data.porcentaje_fraudes,

                    80,

                    95
                ],

                borderColor:"#3b82f6",

                backgroundColor:
                    "rgba(59,130,246,0.2)"
            }]
        },

        options:{

            responsive:true,

            scales:{

                r:{

                    ticks:{
                        color:"white"
                    },

                    pointLabels:{
                        color:"white"
                    },

                    grid:{
                        color:
                            "rgba(255,255,255,0.1)"
                    }
                }
            },

            plugins:{

                legend:{
                    labels:{
                        color:"white"
                    }
                }
            }
        }
    });
}


// ========================================
// EVENTOS
// ========================================

function inicializarEventos(){

    document
        .getElementById(
            "btnLogout"
        )
        .addEventListener(
            "click",
            logout
        );

    document
        .getElementById(
            "btnCargarEstadisticas"
        )
        .addEventListener(
            "click",
            cargarDashboard
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

    activarMenu();

    inicializarEventos();

    await cargarArchivos();
};
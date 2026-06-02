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


// Elementos DOM
const archivoSelect = document.getElementById("archivoSelect");
const loaderDashboard = document.getElementById("loaderDashboard");
const loaderImportar = document.getElementById("loaderImportar");


// Charts
let pieChart = null;
let barChart = null;
let lineChart = null;
let radarChart = null;

// Elimina una gráfica existente antes de recrearla.
function limpiarChart(chart){

    if(chart){

        chart.destroy();
    }

    return null;
}

// Importa un archivo CSV al sistema.
async function importarArchivo(){

    const input =
        document.getElementById(
            "csvFile"
        );

    if(!input.files.length){

        alert(
            "Seleccione un archivo CSV"
        );

        return;
    }

    loaderImportar
        .classList
        .remove("hidden");

    const formData =
        new FormData();

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

        if(response && response.ok){

            await cargarArchivos();
        }

        if(!response){
            return;
        }

        await obtenerData(response);

        alert(
            "Archivo importado correctamente"
        );

        input.value = "";

        // RECARGAR LISTA ARCHIVOS

        await cargarArchivos();

        if(archivoSelect.options.length > 1){

            const ultimoArchivoImportado =

                archivoSelect.options[
                    archivoSelect.options.length - 1
                ].value;

            archivoSelect.value =
                ultimoArchivoImportado;

            localStorage.setItem(
                "ultimoArchivo",
                ultimoArchivoImportado
            );

            await cargarDashboard();
        }

    }catch(error){

        mostrarError(
            "Error importando archivo",
            error
        );

    }finally{

        loaderImportar
            .classList
            .add("hidden");
    }
}

// Carga los archivos importados por el usuario.
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

        let opciones = `
            <option value="">
                Seleccionar archivo
            </option>
        `;

        archivos.forEach(archivo => {

            opciones += `
                <option value="${archivo.id_archivo}">
                    ${archivo.nombre_archivo}
                </option>
            `;
        });

        archivoSelect.innerHTML = opciones;

    }catch(error){

        mostrarError(
            "Error cargando archivos",
            error
        );
    }
}

// Genera el dashboard de estadísticas del archivo seleccionado.
async function cargarDashboard(){

    const idArchivo =
        archivoSelect.value;

    localStorage.setItem(
        "ultimoArchivo",
        idArchivo
    );

    if(!idArchivo){
        limpiarDashboard();
        return;
    }

    // MOSTRAR LOADER

    loaderDashboard
        .classList
        .remove("hidden");

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

        // Ignorar errores al salir de página
        if(
            !document.body.contains(loaderDashboard)
        ){
            return;
        }

        mostrarError(
            "Error generando dashboard",
            error
        );

    }finally{

        if(loaderDashboard){

            loaderDashboard
                .classList
                .add("hidden");
        }
    }
}

// Actualiza los indicadores KPI del dashboard.
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

// Genera la gráfica circular de fraudes y consumos normales.
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

            maintainAspectRatio:true,

            aspectRatio:1,

            cutout:"50%",

            radius:"75%",

            plugins:{

                legend:{
                    labels:{
                        color:"white"
                    }
                }
            },

            animation:{
                duration:1500,
                easing:"easeOutQuart"
            }
        }
    });
}

// Genera la gráfica de barras de consumos medios.
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

            labels:
                data.top_consumos.map(
                    item => item.curva
                ),

            datasets:[{

                label:
                    "Top consumos medios",

                data:
                    data.top_consumos.map(
                        item => item.consumo
                    ),

                backgroundColor:
                    "#3b82f6"
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
            },

            animation:{
                duration:1500,
                easing:"easeOutQuart"
            }
        }
    });
}

// Genera la gráfica de riesgo de fraude.
function crearLineChart(data){

    const ctx =
        document
            .getElementById(
                "graficaLineal"
            );

    lineChart =
        limpiarChart(lineChart);

    lineChart = new Chart(ctx, {

        type:"bar",

        data:{

            labels:
                data.top_riesgo.map(
                    item => item.curva
                ),

            datasets:[{

                label:
                    "Probabilidad fraude (%)",

                data:
                    data.top_riesgo.map(
                        item => item.probabilidad
                    ),

                backgroundColor:
                    "#ef4444"
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

                    beginAtZero:true,

                    ticks:{
                        color:"white"
                    },

                    title:{
                        display:true,
                        text:"Probabilidad fraude (%)",
                        color:"white"
                    }
                },

                x:{

                    ticks:{
                        color:"white"
                    },

                    title:{
                        display:true,
                        text:"Curvas",
                        color:"white"
                    }
                }
            },

            animation:{
                duration:1500,
                easing:"easeOutQuart"
            }
        }
    });
}

// Genera la distribución de probabilidades de fraude.
function crearRadarChart(data){

    const ctx =
        document
            .getElementById(
                "graficaRadar"
            );

    radarChart =
        limpiarChart(radarChart);

    const rangos = [

        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%"
    ];

    const conteos =
        new Array(10).fill(0);

    data.probabilidades.forEach(p => {

        let indice =
            Math.floor(p / 10);

        if(indice >= 10){
            indice = 9;
        }

        conteos[indice]++;
    });

    radarChart = new Chart(ctx, {

        type:"bar",

        data:{

            labels: rangos,

            datasets:[{

                label:
                    "Número de curvas",

                data: conteos,

                backgroundColor:
                    "#3b82f6"
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
                    },

                    title:{
                        display:true,
                        text:"Número de curvas",
                        color:"white"
                    }
                },

                x:{

                    ticks:{
                        color:"white"
                    },

                    title:{
                        display:true,
                        text:"Rango probabilidad fraude",
                        color:"white"
                    }
                }
            },

            animation:{
                duration:1500,
                easing:"easeOutQuart"
            }
        }
    });
}

//Elimina un archivo del usuario
async function eliminarArchivo(){

    const idArchivo = archivoSelect.value;

    if(!idArchivo){
        alert("Selecciona un archivo primero");
        return;
    }

    const nombreArchivo = 
        archivoSelect.options[archivoSelect.selectedIndex].text;

    const confirmado = confirm(
        `¿Seguro que quieres eliminar "${nombreArchivo}"?\nEsta acción no se puede deshacer.`
    );

    if(!confirmado) return;

    try{
        const response = await fetchAuth(
            `/api/operador/archivo/${idArchivo}`,
            { method: "DELETE" }
        );

        if(!response) return;

        await obtenerData(response);

        alert("Archivo eliminado correctamente");

        limpiarDashboard();
        await cargarArchivos();

    }catch(error){
        mostrarError("Error eliminando archivo", error);
    }
}

// Inicializa los eventos de interacción de la página.
function inicializarEventos(){

    document
        .getElementById("btnLogout")
        .addEventListener("click", logout);
    
    archivoSelect.addEventListener("change", cargarDashboard);

    document
        .getElementById("btnImportar")
        .addEventListener("click", importarArchivo);
    
    document
        .getElementById("btnEliminar")
        .addEventListener("click", eliminarArchivo);
}

// Inicializa la pantalla de estadísticas al cargar la página.
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

    const ultimoArchivo =
        localStorage.getItem(
            "ultimoArchivo"
        );

    if(
        ultimoArchivo &&
        [...archivoSelect.options].some(
            option =>
                option.value === ultimoArchivo
        )
    ){

        archivoSelect.value =
            ultimoArchivo;

    }else if(archivoSelect.options.length > 1){

        archivoSelect.value =
            archivoSelect.options[
                archivoSelect.options.length - 1
            ].value;
    }

    if(archivoSelect.value){

        await cargarDashboard();
    }
};

// Limpia los KPIs y destruye las gráficas activas.
function limpiarDashboard(){

    // KPIs

    document.getElementById(
        "kpiTotal"
    ).innerText = "---";

    document.getElementById(
        "kpiFraudes"
    ).innerText = "---";

    document.getElementById(
        "kpiNormales"
    ).innerText = "---";

    document.getElementById(
        "kpiPorcentaje"
    ).innerText = "---";

    // Charts

    pieChart = limpiarChart(pieChart);

    barChart = limpiarChart(barChart);

    lineChart = limpiarChart(lineChart);

    radarChart = limpiarChart(radarChart);
}
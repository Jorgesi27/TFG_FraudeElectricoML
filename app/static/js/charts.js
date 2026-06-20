// Destruye una gráfica existente y libera memoria.
export function limpiarGrafica(chartInstance){

    if(chartInstance){
        chartInstance.destroy();
    }

    return null;
}

export function crearGraficaLineal(ctx, labels, valores, titulo, yMin = null, yMax = null, xMin = null, xMax = null) {

    return new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: titulo,
                data: valores,
                borderColor: "#3b82f6",
                backgroundColor: "rgba(59,130,246,0.1)",
                fill: false,
                pointRadius: 0,
                borderWidth: 1,
                tension: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                zoom: {
                    zoom: {
                        wheel: { enabled: true },
                        pinch: { enabled: true },
                        mode: "x"
                    },
                    pan: {
                        enabled: true,
                        mode: "x"
                    },
                    limits: {
                        x: {
                            min: "original",
                            max: "original",
                            minRange: 100
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: "Hora", color: "white" },
                    ticks: {
                        color: "white",
                        maxTicksLimit: 12
                    },
                    grid: { color: "rgba(255,255,255,0.1)" },
                    min: xMin,
                    max: xMax
                },
                y: {
                    title: { display: true, text: "Consumo [kW]", color: "white" },
                    ticks: { color: "white" },
                    grid: { color: "rgba(255,255,255,0.1)" },
                    min: yMin,
                    max: yMax
                }
            }
        }
    });
}

// Crea una gráfica circular con la distribución de fraudes y consumos normales.
export function crearGraficaPie(
    ctx,
    data
){

    return new Chart(ctx, {

        type: "doughnut",

        data: {

            labels: [
                "Fraudes",
                "Normales"
            ],

            datasets: [{

                data: [
                    data.fraudes,
                    data.normales
                ],

                backgroundColor: [
                    "#ef4444",
                    "#3b82f6"
                ]
            }]
        },

        options: {

            responsive: true,

            maintainAspectRatio: true,

            aspectRatio: 1
        }
    });
}

// Crea una gráfica de barras comparando fraudes y consumos normales.
export function crearGraficaBarras(
    ctx,
    data
){

    return new Chart(ctx, {

        type: "bar",

        data: {

            labels: [
                "Fraudes",
                "Normales"
            ],

            datasets: [{

                label: "Cantidad",

                data: [
                    data.fraudes,
                    data.normales
                ],

                backgroundColor: [
                    "#ef4444",
                    "#3b82f6"
                ]
            }]
        }
    });
}

// Crea una gráfica lineal coloreando en rojo los tramos marcados como fraude.
export function crearGraficaLinealConFraude(ctx, labels, valores, fraudes, titulo) {

    return new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: titulo,
                data: valores,
                borderColor: "#3b82f6",
                backgroundColor: "rgba(59,130,246,0.1)",
                segment: {
                    borderColor: (ctx) => {
                        const idx = ctx.p1DataIndex;
                        return fraudes[idx] === 1 ? "#ef4444" : "#3b82f6";
                    }
                },
                fill: false,
                pointRadius: 0,
                borderWidth: 1.5,
                tension: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                zoom: {
                    zoom: {
                        wheel: { enabled: true },
                        pinch: { enabled: true },
                        mode: "x"
                    },
                    pan: {
                        enabled: true,
                        mode: "x"
                    },
                    limits: {
                        x: {
                            min: "original",
                            max: "original",
                            minRange: 100
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: "Hora", color: "white" },
                    ticks: {
                        color: "white",
                        maxTicksLimit: 12
                    },
                    grid: { color: "rgba(255,255,255,0.1)" }
                },
                y: {
                    title: { display: true, text: "Consumo [kW]", color: "white" },
                    ticks: { color: "white" },
                    grid: { color: "rgba(255,255,255,0.1)" }
                }
            }
        }
    });
}
// Destruye una gráfica existente y libera memoria.
export function limpiarGrafica(chartInstance){

    if(chartInstance){
        chartInstance.destroy();
    }

    return null;
}

// Crea una gráfica lineal para representar curvas de consumo.
export function crearGraficaLineal(
    ctx,
    labels,
    valores,
    titulo
){

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

            interaction: {
                mode: "index",
                intersect: false
            },

            plugins: {

                legend: {
                    display: true
                }
            },

            scales: {

                x: {

                    ticks: {

                        color: "white",

                        autoSkip: true,

                        maxTicksLimit: 20
                    }
                },

                y: {

                    beginAtZero: false
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
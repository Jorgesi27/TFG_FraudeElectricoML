export function limpiarGrafica(chartInstance){

    if(chartInstance){
        chartInstance.destroy();
    }

    return null;
}


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

                backgroundColor: "rgba(59,130,246,0.2)",

                pointRadius: 4,

                pointHoverRadius: 6,

                pointBackgroundColor: "#3b82f6",

                pointBorderColor: "#3b82f6",

                tension: 0.3
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

                        maxTicksLimit: 12,

                        maxRotation: 45,

                        minRotation: 45
                    }
                },

                y: {

                    beginAtZero: false
                }
            }
        }
    });
}


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
// Destruye una gráfica existente y libera memoria.
export function limpiarGrafica(chartInstance){

    if(chartInstance){
        chartInstance.destroy();
    }

    return null;
}

// Crea una gráfica lineal para representar curvas de consumo.
export function crearGraficaLineal(ctx, labels, valores, titulo) {

    // Filtrar pares donde el valor es 0
    const filtrado = labels
        .map((l, i) => ({ x: l, y: valores[i] }))
        .filter(p => p.y !== 0 && p.y !== null);

    return new Chart(ctx, {

        type: "line",

        data: {
            datasets: [{
                label: titulo,
                data: filtrado,
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
                legend: { display: true },
                zoom: {
                    zoom: {
                        wheel: { enabled: true },
                        pinch: { enabled: true },
                        mode: "x"
                    },
                    pan: {
                        enabled: true,
                        mode: "x"
                    }
                }
            },

            scales: {
                x: {
                    type: "linear",
                    title: {
                        display: true,
                        text: "Hora",
                        color: "white"
                    },
                    ticks: { color: "white" },
                    grid: { color: "rgba(255,255,255,0.1)" }
                },
                y: {
                    title: {
                        display: true,
                        text: "Consumo [kW]",
                        color: "white"
                    },
                    ticks: { color: "white" },
                    grid: { color: "rgba(255,255,255,0.1)" }
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
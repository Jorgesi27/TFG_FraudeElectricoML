async function importarArchivo() {

    const input = document.getElementById("csvFile");

    if (!input.files.length) {
        alert("Seleccione un archivo CSV");
        return;
    }

    const formData = new FormData();
    formData.append("archivo", input.files[0]);

    try {

        const response = await fetch(
            "/api/operador/importar-csv",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            alert(data.detail);
            return;
        }

        alert("Archivo importado correctamente");

        cargarArchivos();

    } catch (error) {

        console.error(error);

        alert("Error al importar archivo");
    }
}


async function cargarArchivos() {

    try {

        const response = await fetch(
            "/api/operador/archivos"
        );

        const archivos = await response.json();

        const selector =
            document.getElementById("archivoSelect");

        selector.innerHTML =
            '<option value="">Seleccionar archivo</option>';

        archivos.forEach(archivo => {

            selector.innerHTML += `
                <option value="${archivo.id_archivo}">
                    ${archivo.nombre_archivo}
                </option>
            `;
        });

    } catch (error) {

        console.error(error);

        alert("Error cargando archivos");
    }
}


async function cargarCurvas() {

    const idArchivo =
        document.getElementById("archivoSelect").value;

    if (!idArchivo) {
        return;
    }

    try {

        const response = await fetch(
            `/api/operador/curvas/${idArchivo}`
        );

        const data = await response.json();

        const curvas = data.curvas;

        const selector =
            document.getElementById("curvaSelect");

        selector.innerHTML =
            '<option value="">Seleccionar curva</option>';

        curvas.forEach(curva => {

            selector.innerHTML += `
                <option value="${curva.id_curva}">
                    ${curva.identificador_curva}
                </option>
            `;
        });

    } catch (error) {

        console.error(error);

        alert("Error cargando curvas");
    }
}


let chart = null;

async function mostrarGrafica() {

    const idCurva =
        document.getElementById("curvaSelect").value;

    if (!idCurva) {
        return;
    }

    try {

        const response = await fetch(
            `/api/operador/curva/${idCurva}`
        );

        const curva = await response.json();

        const datos =
            curva.datos_consumo;

        const valores =
            Object.values(datos)
                .filter(v => typeof v === "number");

        const labels =
            valores.map((_, i) => i + 1);

        const ctx =
            document
                .getElementById("graficaCurva")
                .getContext("2d");

        if (chart) {
            chart.destroy();
        }

        chart = new Chart(ctx, {

            type: "line",

            data: {

                labels: labels,

                datasets: [{
                    label: curva.identificador_curva,
                    data: valores
                }]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });

    } catch (error) {

        console.error(error);

        alert("Error mostrando gráfica");
    }
}


async function prediccionHistorica() {

    const input = document.getElementById("csvFile");

    if (!input.files.length) {
        alert("Seleccione un archivo CSV");
        return;
    }

    const formData = new FormData();

    formData.append(
        "archivo",
        input.files[0]
    );

    try {

        const response = await fetch(
            "/api/operador/prediccion/historica",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            alert(data.detail);
            return;
        }

        mostrarResultado(data);

    } catch (error) {

        console.error(error);

        alert("Error en predicción histórica");
    }
}


async function prediccionTiempoReal() {

    const input = document.getElementById("csvFile");

    if (!input.files.length) {
        alert("Seleccione un archivo CSV");
        return;
    }

    const formData = new FormData();

    formData.append(
        "archivo",
        input.files[0]
    );

    try {

        const response = await fetch(
            "/api/operador/prediccion/tiempo-real",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            alert(data.detail);
            return;
        }

        mostrarResultado(data);

    } catch (error) {

        console.error(error);

        alert("Error en predicción tiempo real");
    }
}


function mostrarResultado(data) {

    const resultado =
        document.getElementById("resultado");

    const probabilidad =
        document.getElementById("probabilidad");

    if (!data.predicciones.length) {

        resultado.innerText = "---";
        probabilidad.innerText = "---";

        return;
    }

    const prediccion =
        data.predicciones[0];

    resultado.innerText =
        prediccion.resultado;

    probabilidad.innerText =
        prediccion.probabilidad_fraude;
}


window.onload = () => {

    cargarArchivos();

};
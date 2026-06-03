import {
    fetchAuth,
    obtenerData
} from "./api.js";

import {
    mostrarError
} from "./ui.js";

// Muestra el resultado visual de la predicción.
export function mostrarResultado(
    resultadoEl,
    probabilidadEl,
    data
){
    if(!data || !data.resultado) return;

    if(data.estado === "acumulando"){
        resultadoEl.innerHTML = `
            <div class="badge-analizando fade-in">
                ⏳ Acumulando datos...
            </div>
        `;
        probabilidadEl.innerHTML = "";
        return;
    }

    const esFraude = data.resultado.toLowerCase() === "fraude";

    resultadoEl.innerHTML = esFraude
        ? `<div class="badge-fraude fade-in">🔴 FRAUDE DETECTADO</div>`
        : `<div class="badge-normal fade-in">🟢 CONSUMO NORMAL</div>`;

    probabilidadEl.innerHTML = `
        <p>Probabilidad fraude: ${data.probabilidad}%</p>
    `;
}

// Realiza una predicción histórica de una curva.
export async function prediccionHistorica(
    idCurva
){

    try{

        const response = await fetchAuth(

            `/api/operador/prediccion/historica/${idCurva}`,

            {
                method:"POST"
            }
        );

        if(!response){
            return null;
        }

        return await obtenerData(response);

    }catch(error){

        mostrarError(
            "Error realizando la predicción histórica",
            error
        );

        return null;
    }
}
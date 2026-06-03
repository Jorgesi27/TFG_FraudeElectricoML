import {
    fetchAuth,
    obtenerData
} from "./api.js";

import {
    mostrarError
} from "./ui.js";

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

    // Histórica devuelve probabilidad_fraude y probabilidad_media
    // Stream devuelve probabilidad
    if(data.probabilidad_fraude){
        probabilidadEl.innerHTML = `
            <p>Probabilidad máxima: ${data.probabilidad_fraude}</p>
            <p>Probabilidad media: ${data.probabilidad_media}</p>
        `;
    } else {
        probabilidadEl.innerHTML = `
            <p>Probabilidad fraude: ${data.probabilidad}%</p>
        `;
    }
}

export async function prediccionHistorica(idCurva){

    try{
        const response = await fetchAuth(
            `/api/operador/prediccion/historica/${idCurva}`,
            { method: "POST" }
        );

        if(!response){ return null; }

        return await obtenerData(response);

    }catch(error){
        mostrarError("Error realizando la predicción histórica", error);
        return null;
    }
}
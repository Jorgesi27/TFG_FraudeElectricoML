import {
    fetchAuth,
    obtenerData
} from "./api.js";

import {
    mostrarError
} from "./ui.js";


export function mostrarResultado(
    resultadoHistorico,
    probabilidadHistorica,
    data
){

    resultadoHistorico.innerHTML =
        "---";

    probabilidadHistorica.innerText =
        "---";

    if(
        !data ||
        !data.resultado
    ){
        return;
    }

    const esFraude =

        data.resultado.toLowerCase()
            === "fraude";

    resultadoHistorico.innerHTML =

        esFraude

        ?

        `
        <div class="badge-fraude fade-in">
            🔴 FRAUDE DETECTADO
        </div>
        `

        :

        `
        <div class="badge-normal fade-in">
            🟢 CONSUMO NORMAL
        </div>
        `;

    probabilidadHistorica.innerText =

        `Probabilidad de fraude: ${data.probabilidad_fraude}`;
}


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
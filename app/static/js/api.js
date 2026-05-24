import {
    obtenerHeadersAuth,
    comprobarNoAutorizado
} from "./auth.js";

// Realiza peticiones autenticadas.
export async function fetchAuth(
    url,
    opciones = {}
){

    const response = await fetch(url, {

        ...opciones,

        headers: {

            ...obtenerHeadersAuth(),

            ...(opciones.headers || {})
        }
    });

    if(comprobarNoAutorizado(response)){
        return null;
    }

    return response;
}

// Obtiene y valida peticiones JSON.
export async function obtenerData(response){

    const data = await response.json();

    if(!response.ok){

        throw new Error(
            data.detail || "Error en la petición"
        );
    }

    return data;
}
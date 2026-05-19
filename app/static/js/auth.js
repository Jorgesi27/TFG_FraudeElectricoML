export function obtenerHeadersAuth(){

    const token =
        localStorage.getItem("token");

    return {
        "Authorization":
            `Bearer ${token}`
    };
}


export function limpiarSesion(){

    localStorage.removeItem("token");

    localStorage.removeItem("usuario");
}


export function redirigirLogin(){

    window.location.href =
        "/api/operador/login-page";
}


export function comprobarNoAutorizado(response){

    if(response.status === 401){

        limpiarSesion();

        redirigirLogin();

        return true;
    }

    return false;
}


// ========================================
// LOGOUT
// ========================================

export function logout(){

    limpiarSesion();

    redirigirLogin();
}
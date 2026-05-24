// Recupera el token JWT almacenado.
export function obtenerHeadersAuth(){

    const token =
        localStorage.getItem("token");

    return {
        "Authorization":
            `Bearer ${token}`
    };
}

// Elimina los datos de autenticación.
export function limpiarSesion(){

    localStorage.removeItem("token");

    localStorage.removeItem("usuario");
}

// Envía al usuario a la pantalla login.
export function redirigirLogin(){

    window.location.href =
        "/api/operador/login-page";
}

// Verifica si el token es inválido.
export function comprobarNoAutorizado(response){

    if(response.status === 401){

        limpiarSesion();

        redirigirLogin();

        return true;
    }

    return false;
}

// Cierra la sesión del usuario.
export function logout(){

    limpiarSesion();

    redirigirLogin();
}
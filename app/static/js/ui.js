// Muestra un elemento HTML.
export function mostrarElemento(elemento){
    elemento.style.display = "block";
}

// Oculta un elemento HTML.
export function ocultarElemento(elemento){
    elemento.style.display = "none";
}

// Muestra un mensaje de error por pantalla.
export function mostrarError(mensaje, error = null){

    if(error){
        console.error(error);
    }

    alert(mensaje);
}

// Muestra un elemento usando display flex.
export function mostrarFlex(elemento){
    elemento.style.display = "flex";
}
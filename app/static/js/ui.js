export function mostrarElemento(elemento){
    elemento.style.display = "block";
}

export function ocultarElemento(elemento){
    elemento.style.display = "none";
}

export function mostrarError(mensaje, error = null){

    if(error){
        console.error(error);
    }

    alert(mensaje);
}

export function mostrarFlex(elemento){
    elemento.style.display = "flex";
}
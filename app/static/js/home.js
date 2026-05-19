import {
    fetchAuth,
    obtenerData
} from "./api.js";

import {
    logout,
    redirigirLogin
} from "./auth.js";

import {
    activarMenu
} from "./menu.js";

import {
    mostrarError
} from "./ui.js";


// ========================================
// IMPORTAR CSV
// ========================================

async function importarArchivo(){

    const input =
        document.getElementById(
            "csvFile"
        );

    if(!input.files.length){

        alert(
            "Seleccione un archivo CSV"
        );

        return;
    }

    const loader =
        document.getElementById(
            "loaderImportar"
        );

    loader.style.display = "block";

    const formData =
        new FormData();

    formData.append(
        "archivo",
        input.files[0]
    );

    try{

        const response = await fetchAuth(

            "/api/operador/importar-csv",

            {
                method:"POST",
                body:formData
            }
        );

        if(!response){
            return;
        }

        await obtenerData(response);

        alert(
            "Archivo importado correctamente"
        );

        input.value = "";

    }catch(error){

        mostrarError(
            "Error importando archivo",
            error
        );

    }finally{

        loader.style.display = "none";
    }
}


// ========================================
// EVENTOS
// ========================================

function inicializarEventos(){

    document
        .getElementById(
            "btnLogout"
        )
        .addEventListener(
            "click",
            logout
        );

    document
        .getElementById(
            "btnImportar"
        )
        .addEventListener(
            "click",
            importarArchivo
        );
}


// ========================================
// INIT
// ========================================

window.onload = () => {

    const token =
        localStorage.getItem("token");

    if(!token){

        redirigirLogin();

        return;
    }

    activarMenu();

    inicializarEventos();
};
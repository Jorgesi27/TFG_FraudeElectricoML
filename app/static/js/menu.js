// Activa el menú y la navegación a través de él.
export function activarMenu(){

    const links =
        document.querySelectorAll(".sidebar a");

    const current =
        window.location.pathname;

    links.forEach(link => {

        if(current.includes(link.getAttribute("href"))){

            link.classList.add("active");
        }
    });
}
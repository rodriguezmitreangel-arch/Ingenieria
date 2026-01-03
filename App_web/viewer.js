document.addEventListener("DOMContentLoaded", () => {

    const welcome = document.getElementById("welcome-screen");
    const splash = document.getElementById("splash-screen");

    const btnFullscreen = document.getElementById("btn-fullscreen");
    const btnNormal = document.getElementById("btn-normal");

    // Función para activar Fullscreen
    function activarFullscreen() {
        const docEl = document.documentElement;
        if (docEl.requestFullscreen) docEl.requestFullscreen();
        else if (docEl.webkitRequestFullscreen) docEl.webkitRequestFullscreen();
        else if (docEl.mozRequestFullScreen) docEl.mozRequestFullScreen();
        else if (docEl.msRequestFullscreen) docEl.msRequestFullscreen();
    }

    // Función para INICIAR el splash después de la elección
    function iniciarSplash() {
        welcome.style.display = "none";  // Ocultar menú inicial
        splash.style.display = "flex";   // Mostrar splash

        // =======================
        // LÓGICA SPLASH
        // =======================

        const splashImg = document.getElementById("splash-img");
        const loadFill = document.getElementById("splash-loading-fill");
        const imagenesSplash = ["1.jpg", "2.jpg"]; // <-- usa tus imágenes
        let index = 0;

        loadFill.style.transitionDuration = "10s";
        setTimeout(() => {
            loadFill.style.width = "100%";
        }, 100);

        function siguienteImagen() {
            if (index < imagenesSplash.length - 1) {
                index++;
                splashImg.style.opacity = 0;
                setTimeout(() => {
                    splashImg.src = imagenesSplash[index];
                    splashImg.style.opacity = 1;
                }, 800);
            } else {
                setTimeout(() => {
                    splash.style.opacity = 0;
                    setTimeout(() => splash.remove(), 1200);
                }, 1000);
            }
        }

        setTimeout(siguienteImagen, 5000);
        setTimeout(siguienteImagen, 10000);

        // Después del splash, continúa la página como siempre
        inicializarGaleria();
    }

    // Eventos de los botones iniciales
    btnFullscreen.addEventListener("click", () => {
        activarFullscreen();
        iniciarSplash();
    });

    btnNormal.addEventListener("click", () => {
        iniciarSplash();
    });


    // =========================================
    // GALERIA ORIGINAL (contenedor y botones)
    // =========================================
  // =========================================
function inicializarGaleria() {

    // Mantener viewer y las imágenes (pantalla de carga)
    const viewer = document.createElement("div");
    viewer.id = "viewer-contenedor";

    const img = document.createElement("img");
    img.id = "viewer-img";
    viewer.appendChild(img);

    const categorias = {
        "CARGA": [
            "assets/1.jpg",
            "assets/2.jpg"
        ]
    };

    let imagenActual = 0;
    let imagenesActivas = categorias["CARGA"];
    img.src = imagenesActivas[0];

    // Animación del slideshow
    setInterval(() => {
        imagenActual = (imagenActual + 1) % imagenesActivas.length;
        img.style.opacity = 0;
        setTimeout(() => {
            img.src = imagenesActivas[imagenActual];
            img.style.opacity = 1;
        }, 400);
    }, 6000);

    // Insertar solo el viewer (ya no contenedor ni botones)
    const hero = document.querySelector(".hero");
    hero.insertAdjacentElement("afterend", viewer);

    // Redirigir después de X segundos
    setTimeout(() => {
        window.location.href = "frontend/index.html"; 
    }, 8000); // ⬅ tiempo antes de redirigir (ajústalo)
}});

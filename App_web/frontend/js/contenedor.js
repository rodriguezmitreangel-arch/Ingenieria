document.addEventListener("DOMContentLoaded", () => {

    // ========== CONTENEDOR DE BOTONES ==========
    const contenedor = document.createElement("div");
    contenedor.id = "contenedor-botones";
    contenedor.style.display = "flex";
    contenedor.style.justifyContent = "center";
    contenedor.style.flexWrap = "wrap";
    contenedor.style.gap = "20px";
    contenedor.style.marginTop = "25px";

    // ========== CONTENEDOR DE IMÁGENES ==========
    const viewer = document.createElement("div");
    viewer.id = "viewer-contenedor";
    viewer.style.width = "100%";
    viewer.style.maxWidth = "900px";
    viewer.style.height = "420px";
    viewer.style.margin = "30px auto";
    viewer.style.borderRadius = "20px";
    viewer.style.background = "#11141b";
    viewer.style.boxShadow = "0 0 25px rgba(0,0,0,0.4)";
    viewer.style.overflow = "hidden";
    viewer.style.position = "relative";

    const img = document.createElement("img");
img.id = "viewer-img";

img.style.width = "100%";
img.style.height = "100%";

img.style.objectFit = "contain";   // ajuste inteligente
img.style.background = "#11141b";

// Centrado perfecto
img.style.position = "absolute";
img.style.top = "50%";
img.style.left = "50%";
img.style.transform = "translate(-50%, -50%)";

img.style.transition = "opacity 0.5s ease";


    viewer.appendChild(img);

    // ========== IMÁGENES POR CATEGORÍA (según el botón) ==========
    const categorias = {
        "MODELOS 3D": [
            "assets/1.jpg",
            "assets/1.jpg",
            "assets/1.jpg"
        ],
        "TEXTURAS": [
            "assets/2.jpg",
            "assets/2.jpg",
            "assets/2.jpg"
        ],
        "PLAN BÁSICO": [
            "https://i.imgur.com/4uTgFyp.jpeg",
            "https://i.imgur.com/CAuDFo9.jpeg",
            "https://i.imgur.com/cyOwTQw.jpeg"
        ],
        "PLAN PRO": [
            "https://i.imgur.com/oVBmVuC.jpeg",
            "https://i.imgur.com/XK2fVB0.jpeg",
            "https://i.imgur.com/PtiwYp9.jpeg"
        ],
        "VISION ARTIFICIAL": [
            "https://i.imgur.com/W8oHq1N.jpeg",
            "https://i.imgur.com/ZdQtdut.jpeg",
            "https://i.imgur.com/q0ma2Wc.jpeg"
        ]
    };

    let imagenActual = 0;
    let imagenesActivas = categorias["MODELOS 3D"]; // por defecto
    img.src = imagenesActivas[0];

    // Cambiar imagen automática
    setInterval(() => {
        imagenActual = (imagenActual + 1) % imagenesActivas.length;
        img.style.opacity = 0;
        setTimeout(() => {
            img.src = imagenesActivas[imagenActual];
            img.style.opacity = 1;
        }, 400);
    }, 6000);

    // ========== BOTONES ==========
    const botones = Object.keys(categorias);

    botones.forEach(nombre => {
        const btn = document.createElement("button");
        btn.className = "btn-primary panel-hero";
        btn.textContent = nombre;

        btn.style.cursor = "pointer";

        btn.addEventListener("click", () => {
            imagenesActivas = categorias[nombre];
            imagenActual = 0;
            img.src = imagenesActivas[0];
        });

        contenedor.appendChild(btn);
    });

    // ========== INYECTAR CONTENIDO ==========
    const hero = document.querySelector(".hero");
    hero.insertAdjacentElement("afterend", contenedor);
    contenedor.insertAdjacentElement("afterend", viewer);

});

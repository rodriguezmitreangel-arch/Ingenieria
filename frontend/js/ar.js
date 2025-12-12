document.addEventListener("DOMContentLoaded", () => {

    console.log("AR.js cargado ✔");

    // ========================================
    // ESPERA A QUE EXISTA EL CONTENEDOR AR
    // ========================================
    const arBox = document.querySelector("#view-ar .ar-box");

    if (!arBox) {
        console.error("❌ No existe .ar-box dentro de #view-ar");
        return;
    }

    // Evitar duplicados al cambiar de vistas
    if (document.getElementById("contenedor-ar")) {
        console.log("⚠ contenedor-ar ya existe, no se crea otro.");
        return;
    }

    // ========================================
    // FUNCIÓN PARA ENVIAR COMANDOS AL SERVER
    // ========================================
    async function sendAction(action) {
        console.log(">> Enviando acción:", action);

        try {
            const res = await fetch("http://127.0.0.1:5000/" + action, {
                method: "POST",
                headers: { "Content-Type": "application/json" }
            });

            const data = await res.json();
            console.log("Respuesta backend:", data);

        } catch (err) {
            console.error("❌ Error al conectar con backend:", err);
        }
    }

    // ========================================
    // CREAR CONTENEDOR PRINCIPAL
    // ========================================
    const contenedorAR = document.createElement("div");
    contenedorAR.id = "contenedor-ar";
    contenedorAR.style.display = "flex";
    contenedorAR.style.flexDirection = "column";
    contenedorAR.style.alignItems = "center";
    contenedorAR.style.width = "100%";
    contenedorAR.style.padding = "20px";
    contenedorAR.style.background = "#0d1117";
    contenedorAR.style.borderRadius = "15px";
    contenedorAR.style.boxShadow = "0 0 20px rgba(0,0,0,0.5)";
    contenedorAR.style.marginTop = "20px";

    // ========================================
    // TÍTULO
    // ========================================
    const titulo = document.createElement("h2");
    titulo.innerText = "CONTROL GENERAL";
    titulo.style.color = "white";
    titulo.style.marginBottom = "15px";

    contenedorAR.appendChild(titulo);

    // ========================================
    // BOTONES Y ENDPOINTS
    // ========================================
    const botonesAR = {
        "Iniciar Cámara": "start_camera",
        "Detectar libros": "start_detection",
        "Proyectar modelo": "start_projection",
        "Manipular": "start_manipulation",
        "Detener": "stop_all"
    };

    const contenedorBotones = document.createElement("div");
    contenedorBotones.style.display = "flex";
    contenedorBotones.style.gap = "15px";
    contenedorBotones.style.flexWrap = "wrap";
    contenedorBotones.style.justifyContent = "center";
    contenedorBotones.style.marginBottom = "20px";

    // Crear botones dinámicos
    Object.entries(botonesAR).forEach(([nombre, endpoint]) => {
        const btn = document.createElement("button");

        btn.textContent = nombre;
        btn.className = "btn-primary panel-hero";
        btn.style.padding = "12px 20px";
        btn.style.borderRadius = "10px";
        btn.style.cursor = "pointer";

        btn.addEventListener("click", () => {
            sendAction(endpoint);
        });

        contenedorBotones.appendChild(btn);
    });

    contenedorAR.appendChild(contenedorBotones);

    // ========================================
    // VIDEO STREAM
    // ========================================
    const video = document.createElement("img");
    video.id = "ar-stream";
    video.src = "http://127.0.0.1:5000/video_feed";
    video.style.width = "100%";
    video.style.borderRadius = "10px";
    video.style.boxShadow = "0 0 15px rgba(0,0,0,0.4)";

    contenedorAR.appendChild(video);

    // ========================================
    // INSERTAR CONTENEDOR DENTRO DE .ar-box
    // ========================================
    arBox.innerHTML = ""; // limpiar texto anterior
    arBox.appendChild(contenedorAR);

    console.log("✔ Contenedor AR insertado correctamente en .ar-box");
});

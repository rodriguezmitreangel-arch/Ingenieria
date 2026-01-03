document.addEventListener("DOMContentLoaded", () => {

    // ============================
    // REFERENCIAS A LAS VISTAS
    // ============================
    const viewHome = document.getElementById("view-home");
    const viewModelos = document.getElementById("view-modelos");
    const viewAR = document.getElementById("view-ar");
    const viewConfig = document.getElementById("view-config");
    const viewLicencia = document.getElementById("view-licencia");

    const menuItems = document.querySelectorAll(".menu-item");

    // ============================
    // VALIDAR SESIÓN
    // ============================
    const user_id = localStorage.getItem("user_id");

    if (!user_id) {
        window.location.href = "login.html";
        return;
    }

    cargarUsuario(user_id);
    cargarPlanDashboard(user_id);

    // ============================
    // CAMBIO DE VISTAS
    // ============================
    menuItems.forEach(item => {
        item.addEventListener("click", () => {

            // Si es logout, salir directo
            if (item.id === "btn-logout") {
                cerrarSesion();
                return;
            }

            menuItems.forEach(i => i.classList.remove("active"));
            item.classList.add("active");

            const id = item.id;

            ocultarVistas();

            if (id === "btn-home") viewHome.classList.add("visible");
            if (id === "btn-modelos") viewModelos.classList.add("visible");

            // ================================
            // 🔥 Cargar AR.js dinámicamente
            // ================================
            if (id === "btn-ar") {
                viewAR.classList.add("visible");

                // Si ar.js no ha sido cargado antes...
                if (!document.getElementById("script-ar")) {
                    const s = document.createElement("script");
                    s.src = "assets/js/ar.js";
                    s.id = "script-ar";
                    document.body.appendChild(s);
                    console.log("⚡ ar.js cargado dinámicamente");
                }
            }

            if (id === "btn-config") viewConfig.classList.add("visible");

            if (id === "btn-licencia") {
                viewLicencia.classList.add("visible");
                cargarLicencia(user_id);
            }
        });
    });


    // Ocultar todas las vistas
    function ocultarVistas() {
        viewHome.classList.remove("visible");
        viewModelos.classList.remove("visible");
        viewAR.classList.remove("visible");
        viewConfig.classList.remove("visible");
        viewLicencia.classList.remove("visible");
    }

    // ============================
    // CARGAR DATOS DEL USUARIO
    // ============================
    async function cargarUsuario(id) {
        const res = await fetch(`http://127.0.0.1:8000/user-info/${id}`);
        const data = await res.json();

        if (data.status !== "ok") return;

        const user = data.user;

        document.getElementById("user-name").innerText = user.nombre_completo;
        document.getElementById("user-nick").innerText = user.usuario;
        document.getElementById("user-mail").innerText = user.correo;
        document.getElementById("user-id").innerText = user.id;
    }

    // ============================
    // PLAN RESUMIDO DEL DASHBOARD
    // ============================
    async function cargarPlanDashboard(id) {
        const res = await fetch(`http://127.0.0.1:8000/licencia/${id}`);
        const data = await res.json();

        document.getElementById("stat-plan").innerText = data.plan || "FREE";
    }

    // ============================
    // RECUADROS DE LICENCIA
    // ============================
    async function cargarLicencia(id) {

        const res = await fetch(`http://127.0.0.1:8000/licencia/${id}`);
        const data = await res.json();

        const boxActual = document.getElementById("lic-recuadro-actual");
        const boxPro = document.getElementById("lic-recuadro-pro");

        // RECUADRO PLAN ACTUAL
        boxActual.innerHTML = `
            <h2 class="lic-title ${data.plan === 'PRO' ? 'pro' : 'free'}">
                Plan Actual: ${data.plan}
            </h2>

            <ul class="lic-list">
                ${
                    data.plan === "PRO"
                        ? `
                            <li>✔ Modelos ilimitados</li>
                            <li>✔ Cámara seleccionable</li>
                            <li>✔ Texturas ilimitadas</li>
                            <li>✔ Todas las funciones AR premium</li>
                        `
                        : `
                            <li>✔ Modelos precargados</li>
                            <li>✔ Cámara default</li>
                            <li>✔ Sin texturas avanzadas</li>
                            <li>✘ Funciones AR premium bloqueadas</li>
                        `
                }
            </ul>

            <p><b>Fecha activación:</b> ${data.fecha}</p>
            <p><b>Días restantes:</b> ${data.dias}</p>
        `;

        if (data.plan !== "PRO") {
            boxPro.classList.remove("hidden");
        } else {
            boxPro.classList.add("hidden");
        }
    }

    // ============================
    // LOGOUT
    // ============================
    function cerrarSesion() {
        localStorage.removeItem("user_id");
        window.location.href = "login.html";
    }

});

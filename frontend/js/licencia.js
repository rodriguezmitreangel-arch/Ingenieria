document.addEventListener("DOMContentLoaded", () => {
    cargarLicencia();

    const btnComprar = document.getElementById("btn-comprar");
    if (btnComprar) {
        btnComprar.addEventListener("click", comprarPlan);
    }
});

// =====================================================
// CARGAR INFORMACIÓN DE LICENCIA (FUNCIONA PARA AMBAS VISTAS)
// =====================================================
async function cargarLicencia() {

    const id_usuario = localStorage.getItem("id_usuario");

    // Detectar si es licencia.html
    const boxLicenciaHTML = document.getElementById("licencia-info");

    // Detectar si es dashboard.html
    const boxActualDashboard = document.getElementById("lic-recuadro-actual");
    const boxProDashboard = document.getElementById("lic-recuadro-pro");

    if (!id_usuario) {
        if (boxLicenciaHTML) boxLicenciaHTML.innerHTML = "<p>No has iniciado sesión.</p>";
        return;
    }

    try {
        const resp = await fetch(`http://127.0.0.1:8000/api/licencia/${id_usuario}`);
        const data = await resp.json();

        if (!data.ok) {
            if (boxLicenciaHTML) boxLicenciaHTML.innerHTML = "<p>Error al cargar la licencia.</p>";
            return;
        }

        const { plan, fecha_expira, fecha, dias } = data;
        const badgeClass = plan.toLowerCase();

        // =========================================================
        // 1) MODO LICENCIA.HTML
        // =========================================================
        if (boxLicenciaHTML) {
            boxLicenciaHTML.innerHTML = `
                <span class="badge ${badgeClass}">${plan.toUpperCase()}</span>

                <div class="info-line"><b>ID de usuario:</b> ${id_usuario}</div>
                <div class="info-line"><b>Plan actual:</b> ${plan}</div>
                <div class="info-line"><b>Expira el:</b> ${fecha_expira}</div>

                <div class="accesos">
                    ${textoAcceso(plan)}
                </div>
            `;
        }

        // =========================================================
        // 2) MODO DASHBOARD.HTML
        // =========================================================
        if (boxActualDashboard) {

            boxActualDashboard.innerHTML = `
                <h2 class="lic-title ${plan === 'PRO' ? 'pro' : 'free'}">
                    Plan Actual: ${plan}
                </h2>

                <ul class="lic-list">
                    ${
                        plan === "PRO"
                        ? `
                            <li>✔ Modelos ilimitados</li>
                            <li>✔ Cámara seleccionable</li>
                            <li>✔ Texturas ilimitadas</li>
                            <li>✔ Funciones AR Premium</li>
                        `
                        : `
                            <li>✔ Modelos precargados</li>
                            <li>✔ Cámara default</li>
                            <li>✔ Sin texturas avanzadas</li>
                            <li>✘ Funciones AR Premium bloqueadas</li>
                        `
                    }
                </ul>

                <p><b>Fecha activación:</b> ${fecha}</p>
                <p><b>Días restantes:</b> ${dias}</p>
            `;

            // Mostrar recuadro PRO solo si NO es PRO
            if (boxProDashboard) {
                if (plan !== "PRO") {
                    boxProDashboard.classList.remove("hidden");
                } else {
                    boxProDashboard.classList.add("hidden");
                }
            }
        }

    } catch (error) {

        if (boxLicenciaHTML) {
            boxLicenciaHTML.innerHTML = "<p>Error de conexión.</p>";
        }
    }
}

// =====================================================
// TEXTO SEGÚN PLAN
// =====================================================
function textoAcceso(plan) {
    switch (plan) {
        case "FREE": return "Acceso limitado a funciones básicas.";
        case "BASIC": return "Acceso a funciones esenciales del sistema.";
        case "PRO": return "Acceso completo a casi todo el sistema.";
        case "PREMIUM": return "Acceso total + características avanzadas.";
    }
}

// =====================================================
// COMPRAR / ACTUALIZAR PLAN
// =====================================================
async function comprarPlan() {
    const id_usuario = localStorage.getItem("id_usuario");
    if (!id_usuario) {
        alert("Inicia sesión primero.");
        return;
    }

    const plan = prompt("Ingresa el plan a comprar (FREE/BASIC/PRO/PREMIUM):");

    if (!plan) return;

    try {
        const resp = await fetch(`http://127.0.0.1:8000/api/comprar-plan`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                id_usuario: id_usuario,
                plan: plan.toUpperCase()
            })
        });

        const data = await resp.json();

        if (data.ok) {
            alert("Plan actualizado correctamente.");
            cargarLicencia();
        } else {
            alert(data.msg);
        }

    } catch (err) {
        alert("Error conectando al servidor.");
    }
}

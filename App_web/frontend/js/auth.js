document.addEventListener("DOMContentLoaded", () => {

    const user_id = localStorage.getItem("user_id");

    const loginBtn = document.getElementById("login-btn");
    const logoutBtn = document.getElementById("logout-btn");
    const userLabel = document.getElementById("user-label");

    const btnLoginHero = document.getElementById("btnLoginHero");
    const btnRegisterHero = document.getElementById("btnRegisterHero");
    const btnDashboardHero = document.getElementById("btnDashboardHero");

    // ================================
    // SI NO HAY SESIÓN
    // ================================
    if (!user_id) {
        logoutBtn.classList.add("hidden");
        userLabel.classList.add("hidden");
        btnDashboardHero.classList.add("hidden");
        return;
    }

    // ================================
    // SI HAY SESIÓN → mostrar datos
    // ================================
    fetch(`http://127.0.0.1:8000/user-info/${user_id}`)
        .then(r => r.json())
        .then(data => {

            if (data.status !== "ok") return;

            const user = data.user;

            // Mostrar nombre
            userLabel.textContent = `Bienvenido, ${user.usuario}`;
            userLabel.classList.remove("hidden");

            // Ocultar botones de login
            loginBtn.classList.add("hidden");
            btnLoginHero.classList.add("hidden");
            btnRegisterHero.classList.add("hidden");

            // Mostrar botones de cerrar sesión y dashboard
            logoutBtn.classList.remove("hidden");
            btnDashboardHero.classList.remove("hidden");
        });

    // ================================
    // CERRAR SESIÓN
    // ================================
    logoutBtn.addEventListener("click", () => {
        localStorage.removeItem("user_id");
        window.location.href = "index.html";
    });

});


document.getElementById("btnLogin").addEventListener("click", (e) => {
    e.preventDefault(); 
    login();
});

async function login() {
    const usuario = document.getElementById("userLogin").value.trim();
    const password = document.getElementById("passLogin").value.trim();

    if (!usuario || !password) {
        alert("Completa todos los campos.");
        return;
    }

    const res = await fetch("http://127.0.0.1:8000/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({usuario, password})
    });

    const data = await res.json();

    if (data.status !== "ok") {
        alert(data.message);
        return;
    }

    // Guardar ID en localStorage
    localStorage.setItem("user_id", data.user_id);
    window.location.href = "dashboard.html";

}

async function cargarPanel(user_id) {
    const res = await fetch(`http://127.0.0.1:8000/user-info/${user_id}`);
    const data = await res.json();

    if (data.status !== "ok") {
        alert("Error al obtener datos");
        return;
    }

    const user = data.user;

    // Mostrar panel
    document.getElementById("loginBox").style.display = "none";
    document.getElementById("panelUsuario").style.display = "block";

    document.getElementById("tituloBienvenida").innerText = `Hola ${user.nombre_completo}, bienvenido`;
    document.getElementById("infoUsuario").innerText = user.usuario;
    document.getElementById("infoCorreo").innerText = user.correo;
    document.getElementById("infoID").innerText = user.id;
}

function logout() {
    localStorage.removeItem("user_id");
    location.reload();
}

// Si ya estaba logueado antes
window.onload = () => {
    const uid = localStorage.getItem("user_id");
    if (uid) cargarPanel(uid);
};

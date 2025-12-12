document.addEventListener("DOMContentLoaded", () => {

    const form = document.getElementById("formRegister");  // FIX ✔
    const loader = document.getElementById("loader");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const nombre = document.getElementById("nombre").value;
        const usuario = document.getElementById("usuario").value;
        const correo = document.getElementById("correo").value;
        const pass = document.getElementById("password").value;
        const pass2 = document.getElementById("password2").value;

        if (!nombre || !usuario || !correo || !pass || !pass2) {
            alert("Completa todos los campos.");
            return;
        }

        if (pass !== pass2) {
            alert("Las contraseñas no coinciden.");
            return;
        }

        loader.classList.remove("hidden");

        const data = {
            nombre: nombre,
            usuario: usuario,
            correo: correo,
            password: pass
        };

        try {
            const res = await fetch("http://127.0.0.1:8000/register", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(data)
            });

            const json = await res.json();
            alert(json.message || "Registro exitoso");
            loader.classList.add("hidden");
        } catch (err) {
            alert("Error en el servidor");
            loader.classList.add("hidden");
        }

    });
});

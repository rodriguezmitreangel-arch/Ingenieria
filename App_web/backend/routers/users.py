from fastapi import APIRouter
from models import create_user, get_user_by_username
from auth import hash_password, verify_password, create_token

router = APIRouter(prefix="/users", tags=["Usuarios"])

# ----------------------
# REGISTRO
# ----------------------
@router.post("/register")
def register(data: dict):
    nombre = data["nombre"]
    usuario = data["usuario"]
    correo = data["correo"]
    password = data["password"]

    if get_user_by_username(usuario):
        return {"error": "Usuario ya existe"}

    hashed = hash_password(password)

    create_user(nombre, usuario, correo, hashed)

    return {"status": "ok", "message": "Usuario creado correctamente"}


# ----------------------
# LOGIN
# ----------------------
@router.post("/login")
def login(data: dict):
    user = data["usuario"]
    pwd  = data["password"]

    row = get_user_by_username(user)

    if not row:
        return {"error": "Usuario no encontrado"}

    if not verify_password(pwd, row["password_hash"]):
        return {"error": "Contraseña incorrecta"}

    token = create_token(row["id"])

    return {
        "status": "ok",
        "token": token,
        "id_usuario": row["id"]
    }

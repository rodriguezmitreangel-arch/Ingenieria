from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mysql.connector
import bcrypt
import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def conectar_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Admin",
        database="sistema_ar"
    )

class UserRegister(BaseModel):
    nombre: str
    usuario: str
    correo: str
    password: str

class UserLogin(BaseModel):
    usuario: str
    password: str


@app.post("/register")
def register(data: UserRegister):
    conn = conectar_db()
    cur = conn.cursor()

    hashed = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()

    try:
        sql = """
        INSERT INTO usuarios (nombre_completo, usuario, correo, password_hash)
        VALUES (%s, %s, %s, %s)
        """
        cur.execute(sql, (data.nombre, data.usuario, data.correo, hashed))
        conn.commit()

        nuevo_id = cur.lastrowid

        # Crear licencia FREE automáticamente
        sql2 = """
        INSERT INTO licencias (user_id, plan, precio, fecha_activacion, fecha_expiracion)
        VALUES (%s, %s, %s, %s, %s)
        """
        ahora = datetime.datetime.now()
        expiracion = ahora + datetime.timedelta(days=30)

        cur.execute(sql2, (nuevo_id, "FREE", 0, ahora, expiracion))
        conn.commit()

        return {"status": "ok", "message": "Usuario creado"}

    except mysql.connector.Error as e:
        return {"status": "error", "message": str(e)}

    finally:
        cur.close()
        conn.close()


@app.post("/login")
def login(data: UserLogin):
    conn = conectar_db()
    cur = conn.cursor(dictionary=True)

    try:
        sql = "SELECT * FROM usuarios WHERE usuario = %s"
        cur.execute(sql, (data.usuario,))
        user = cur.fetchone()

        if not user:
            return {"status": "error", "message": "Usuario no existe"}, 401

        if not bcrypt.checkpw(data.password.encode(), user["password_hash"].encode()):
            return {"status": "error", "message": "Contraseña incorrecta"}, 401

        return {"status": "ok", "user_id": user["id"]}

    finally:
        cur.close()
        conn.close()



@app.get("/user-info/{user_id}")
def user_info(user_id: int):
    conn = conectar_db()
    cur = conn.cursor(dictionary=True)

    sql = "SELECT id, nombre_completo, usuario, correo FROM usuarios WHERE id = %s"
    cur.execute(sql, (user_id,))
    user = cur.fetchone()

    cur.close()
    conn.close()

    if not user:
        return {"status": "error", "message": "Usuario no encontrado"}

    return {"status": "ok", "user": user}



# LICENCIA REAL

@app.get("/licencia/{user_id}")
def licencia(user_id: int):
    conn = conectar_db()
    cur = conn.cursor(dictionary=True)

    sql = "SELECT * FROM licencias WHERE user_id = %s ORDER BY fecha_activacion DESC LIMIT 1"
    cur.execute(sql, (user_id,))
    lic = cur.fetchone()

    cur.close()
    conn.close()

    if not lic:
        return {"status": "ok", "plan": "FREE", "fecha": "N/A", "dias": "--"}

    hoy = datetime.datetime.now()
    exp = lic["fecha_expiracion"]

    dias_restantes = (exp - hoy).days
    dias_restantes = max(dias_restantes, 0)

    return {
        "status": "ok",
        "plan": lic["plan"],
        "fecha": str(lic["fecha_activacion"]),
        "dias": dias_restantes
    }

from database import get_db

def create_user(nombre, usuario, correo, password_hash):
    conn = get_db()
    cursor = conn.cursor()

    sql = """
        INSERT INTO usuarios (nombre_completo, usuario, correo, password_hash)
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(sql, (nombre, usuario, correo, password_hash))
    conn.commit()

    cursor.close()
    conn.close()


def get_user_by_username(usuario):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM usuarios WHERE usuario=%s", (usuario,)
    )
    row = cursor.fetchone()

    cursor.close()
    conn.close()
    return row


def get_license(id_usuario):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT plan FROM usuarios WHERE id=%s", (id_usuario,)
    )
    row = cursor.fetchone()

    cursor.close()
    conn.close()
    return row["plan"] if row else None


def update_license(id_usuario, nuevo_plan):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE usuarios SET plan=%s WHERE id=%s",
        (nuevo_plan, id_usuario)
    )
    conn.commit()

    cursor.close()
    conn.close()

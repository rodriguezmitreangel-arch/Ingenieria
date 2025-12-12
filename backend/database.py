import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

# Pool de conexiones para alto rendimiento
connection_pool = pooling.MySQLConnectionPool(
    pool_name="ar_pool",
    pool_size=10,
    **DB_CONFIG
)

def get_db():
    try:
        conn = connection_pool.get_connection()
        return conn
    except Exception as e:
        print("Error DB:", e)
        return None

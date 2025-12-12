import bcrypt
import jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = os.getenv("JWT_ALGORITHM")

# -------------------------
# HASH PASSWORD
# -------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

# -------------------------
# GENERAR TOKEN
# -------------------------
def create_token(user_id: int):
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=12)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)

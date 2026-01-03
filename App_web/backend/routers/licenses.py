from fastapi import APIRouter
from models import get_license, update_license

router = APIRouter(prefix="/licenses", tags=["Licencias"])

@router.get("/{id_usuario}")
def estado_licencia(id_usuario: int):
    plan = get_license(id_usuario)
    return {"plan": plan}

@router.post("/comprar/{id_usuario}")
def comprar(id_usuario: int, data: dict):
    plan = data["plan"]
    update_license(id_usuario, plan)
    return {"status": "ok", "message": "Plan actualizado"}

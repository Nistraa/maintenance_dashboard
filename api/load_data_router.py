from fastapi import APIRouter, Depends
from settings import get_db

router = APIRouter()

@router.get("/")
async def read_root():
    return "Router is running"



from fastapi import FastAPI
from models import setup_database
from settings import DATABASE_ENGINE
from api import load_data_router
import logging
import uvicorn

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

logger.info("Setting up Application...")
setup_database(DATABASE_ENGINE)

app = FastAPI()

app.include_router(load_data_router.router, prefix="/load_data", tags=["Load Data"])




@app.get("/")
async def read_root():
    return "Server is running"


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000)


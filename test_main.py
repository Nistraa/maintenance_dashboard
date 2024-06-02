from fastapi.testclient import TestClient
from main import app
from settings import get_db, get_testing_db, DATABASE_TESTING_ENGINE
from models import setup_database

client = TestClient(app)
setup_database(DATABASE_TESTING_ENGINE)
app.dependency_overrides[get_db] = get_testing_db

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Server is running"
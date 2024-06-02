from fastapi.testclient import TestClient
from api.load_data_router import router



client = TestClient(router)

def test_journal_router():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Router is running"
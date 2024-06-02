from fastapi.testclient import TestClient
from api.load_data_router import router



client = TestClient(router)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Server is running"
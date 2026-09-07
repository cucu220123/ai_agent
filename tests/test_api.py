from fastapi.testclient import TestClient

from app.api import app


def test_api_read_endpoints():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/graph/summary").status_code == 200
    assert client.get("/plugins").status_code == 200
    assert client.get("/knowledge/search?q=客户流失").status_code == 200
    assert client.get("/ui").status_code == 200
    assert client.get("/runs").status_code == 200
    assert client.post("/knowledge/ingest", json={"path": "data/business_material.md", "provider": "mock"}).status_code == 200

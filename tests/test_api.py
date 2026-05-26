from fastapi.testclient import TestClient

from misinfo_detector.api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint_loads_training_config() -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "clip_model": "openai/clip-vit-base-patch32",
    }

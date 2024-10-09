import os
import pytest
from fastapi.testclient import TestClient
from src.app import app, MODEL_PATH

MOCK_MODEL_PATH = "tests/mock_model.keras"

@pytest.fixture(scope="session", autouse=True)
def setup_mock_model():
    if not os.path.exists(MOCK_MODEL_PATH):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        model = models.Sequential([
            layers.Input(shape=(150, 150, 3)),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid")
        ])
        model.save(MOCK_MODEL_PATH)

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("src.app.MODEL_PATH", MOCK_MODEL_PATH)
    with TestClient(app) as client:
        yield client

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_predict_success(client):
    import base64
    from PIL import Image as PILImage
    from io import BytesIO

    img = PILImage.new("RGB", (150, 150), color="black")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = client.post("/predict/", json={"image": img_str})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)

def test_predict_invalid_image(client):
    invalid_base64 = "invalid_base64_string"
    response = client.post("/predict/", json={"image": invalid_base64})
    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed"

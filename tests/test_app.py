import pytest
from unittest.mock import patch
from starlette.testclient import TestClient

from app import app


class CompatibleTestClient(TestClient):
    """Custom TestClient to bridge compatibility across all starlette and httpx versions."""
    def __init__(self, app):
        try:
            super().__init__(app)
        except TypeError:
            import httpx
            super(TestClient, self).__init__(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver"
            )


@pytest.fixture
def client():
    """Create a test client instance."""
    return CompatibleTestClient(app)


@pytest.fixture
def valid_payload():
    """Valid prediction request payload with all required 13 features."""
    return {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }


def test_root_endpoint(client):
    """Test GET / returns expected API metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Heart Disease Prediction API"
    assert data["version"] == "1.0.0"


def test_health_check_endpoint(client):
    """Test GET /health returns service status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "preprocessor_loaded" in data


def test_metrics_endpoint(client):
    """Test GET /metrics returns Prometheus format metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "heart_disease_api_requests_total" in response.text


@patch("app.model")
@patch("app.preprocessor")
def test_predict_success(mock_preprocessor, mock_model, client, valid_payload):
    """Test POST /predict with valid payload returns prediction."""
    mock_preprocessor.transform.return_value = [[0.1] * 13]
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["probability"] == 0.8
    assert data["risk_level"] == "High"
    assert "timestamp" in data


@patch("app.model")
@patch("app.preprocessor")
def test_predict_missing_fields(mock_preprocessor, mock_model, client):
    """Test POST /predict with missing required fields raises HTTP 400."""
    incomplete_payload = {"age": 63, "sex": 1}
    response = client.post("/predict", json=incomplete_payload)

    assert response.status_code == 400
    assert "Missing fields" in response.text


@patch("app.model")
@patch("app.preprocessor")
def test_predict_invalid_data_types(mock_preprocessor, mock_model, client, valid_payload):
    """Test POST /predict with non-numeric fields raises HTTP 400."""
    valid_payload["age"] = "invalid_number"
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 400
    assert "Invalid input values" in response.text


@patch("app.model", None)
@patch("app.preprocessor", None)
def test_predict_model_not_loaded(client, valid_payload):
    """Test POST /predict returns HTTP 500 when model or preprocessor is not loaded."""
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 500
    assert "Model not loaded" in response.text
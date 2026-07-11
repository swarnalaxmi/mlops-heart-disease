"""Starlette application for heart disease prediction."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure structured logging
formatter = logging.Formatter(
    '{"timestamp":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

# Load model and preprocessor paths
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.pkl"
PREPROCESSOR_PATH = Path(__file__).parent.parent / "models" / "preprocessor.pkl"

model = None
preprocessor = None

# Prometheus metrics
prediction_counter = Counter(
    "heart_disease_predictions_total",
    "Total number of heart disease predictions generated"
)
prediction_probability = Gauge(
    "heart_disease_prediction_probability",
    "Latest heart disease prediction probability"
)
model_loaded_gauge = Gauge(
    "heart_disease_model_loaded",
    "Whether the model is loaded and available for prediction"
)
model_info_gauge = Gauge(
    "heart_disease_model_info",
    "Static labels for loaded model metadata",
    ["model_name", "model_version"]
)
api_request_counter = Counter(
    "heart_disease_api_requests_total",
    "Count of API requests received",
    ["method", "endpoint", "http_status"]
)
api_request_latency = Histogram(
    "heart_disease_api_latency_seconds",
    "Response latency for API endpoints in seconds",
    ["method", "endpoint"]
)


class PrometheusMetricsMiddleware:
    """Pure ASGI Middleware to record Prometheus metrics safely without stream issues."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = 500  # Default fallback if an unhandled error occurs

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.time() - start_time
            endpoint = scope.get("path", "")
            method = scope.get("method", "")

            api_request_latency.labels(method=method, endpoint=endpoint).observe(elapsed)
            api_request_counter.labels(
                method=method,
                endpoint=endpoint,
                http_status=str(status_code)
            ).inc()


async def load_model_on_startup():
    """Load model and preprocessor safely on startup."""
    global model, preprocessor
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
            model_loaded_gauge.set(1)
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
            model_loaded_gauge.set(0)

        if PREPROCESSOR_PATH.exists():
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
        else:
            logger.warning(f"Preprocessor file not found at {PREPROCESSOR_PATH}")

        model_info_gauge.labels(model_name="best_model", model_version="1.0.0").set(1 if model else 0)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded_gauge.set(0)


def validate_prediction_input(data: dict) -> List:
    required_fields = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal"
    ]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise HTTPException(status_code=400, detail=f"Missing fields: {', '.join(missing_fields)}")

    try:
        return [
            float(data["age"]),
            int(data["sex"]),
            int(data["cp"]),
            float(data["trestbps"]),
            float(data["chol"]),
            int(data["fbs"]),
            int(data["restecg"]),
            float(data["thalach"]),
            int(data["exang"]),
            float(data["oldpeak"]),
            int(data["slope"]),
            int(data["ca"]),
            int(data["thal"]),
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input values: {e}")


async def root(request: Request):
    """Root endpoint."""
    return JSONResponse({
        "message": "Heart Disease Prediction API",
        "version": "1.0.0"
    })


async def health_check(request: Request):
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    })


async def predict(request: Request):
    """Predict heart disease risk."""
    if model is None or preprocessor is None:
        logger.error("Model or preprocessor not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        validated_values = validate_prediction_input(data)

        feature_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope",
            "ca", "thal"
        ]
        features_df = pd.DataFrame([validated_values], columns=feature_names)
        features_processed = preprocessor.transform(features_df)

        prediction = model.predict(features_processed)[0]
        probability = model.predict_proba(features_processed)[0][1]

        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        logger.info(
            f"Prediction made - Input: age={data.get('age')}, Prediction: {prediction}, Probability: {probability:.4f}"
        )

        prediction_counter.inc()
        prediction_probability.set(float(probability))

        return JSONResponse({
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "timestamp": datetime.utcnow().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_metrics(request: Request):
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# App initialization and route definitions
app = Starlette(
    debug=False,
    routes=[
        Route("/", root, methods=["GET"]),
        Route("/health", health_check, methods=["GET"]),
        Route("/predict", predict, methods=["POST"]),
        Route("/metrics", get_metrics, methods=["GET"]),
    ],
)

# Register pure ASGI middleware
app.add_middleware(PrometheusMetricsMiddleware)

# Startup handler
app.add_event_handler("startup", load_model_on_startup)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
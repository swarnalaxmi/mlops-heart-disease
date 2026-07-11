"""Configuration module for MLOps pipeline."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Data paths
RAW_DATA_PATH = DATA_DIR / "heart_disease.csv"
PROCESSED_DATA_PATH = DATA_DIR / "heart_disease_cleaned.csv"

# Model paths
MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "heart-disease-prediction"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# Model thresholds
LOW_RISK_THRESHOLD = 0.4
HIGH_RISK_THRESHOLD = 0.7

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
TESTS_DIR.mkdir(parents=True, exist_ok=True)

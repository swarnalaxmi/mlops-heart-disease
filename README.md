# Heart Disease Prediction - MLOps Project

## Project Overview

An end-to-end Machine Learning Operations (MLOps) project for predicting heart disease risk using the UCI Heart Disease dataset. This project demonstrates modern ML best practices including data science, experiment tracking, CI/CD automation, containerization, Kubernetes orchestration, and API deployment.

## Dataset

**Heart Disease UCI Dataset**
- Source: UCI Machine Learning Repository
- Features: 14+ medical attributes (age, sex, blood pressure, cholesterol, etc.)
- Target: Binary classification (presence/absence of heart disease)
- Download: Automated via `data_processor.py`

## Project Structure

```
mlops-heart-disease/
├── data/                          # Dataset directory
├── models/                        # Trained models and preprocessors
├── src/
│   ├── app.py                    # FastAPI application
│   ├── data_processor.py         # Data loading & preprocessing
│   └── model_trainer.py          # Model training & evaluation
├── tests/                         # Unit tests
│   ├── test_app.py
│   └── test_data_processor.py
├── k8s/                          # Kubernetes manifests
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/            # GitHub Actions CI/CD
│   └── pipeline.yml
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── .gitignore
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd mlops-heart-disease
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset & Train Model

EDA: notebooks\01_EDA.ipynb
This does:
- Histograms
- Correlation heatmaps
- Class distribution plots
- Missing value analysis
- Feature relationship analysis

```bash
python src/model_trainer.py
```

This will:
- Download the Heart Disease UCI Dataset
- Preprocess the data (handle missing values, scale features)
- Train Logistic Regression, Random Forest models, XGBoost, SVM models
- With hyperparamter tuning using GridSearchCV, RandomizedSearchCV
- Log experiments to MLflow
- Save the best model and preprocessor

### 3. Start MLflow UI

```bash
mlflow ui
```

Visit `http://localhost:5000` to view experiment tracking.

MLflow helps track:
• Experiments
• Parameters
• Metrics
• Model versions
• Artifacts
This ensures reproducibility and proper experiment management.

LOGS:
• Model parameters
• Accuracy metrics
• ROC curves/plots
• Confusion matrix
• Saved models

### 4. Run FastAPI Server

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```
### 5. Test the API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": 1,
    "cp": 0,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 100,
    "exang": 0,
    "oldpeak": 0.5,
    "slope": 1,
    "ca": 0,
    "thal": 1
  }'
```

## Model Training

### Models Trained
- **Logistic Regression**: Baseline model for interpretability
- **Random Forest**: Ensemble model for better performance
- **XgBoot**: Ensemble model for better performance
- **SVM**: SVM model for binary classification


### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Experiment Tracking
All experiments are logged to MLflow including:
- Model parameters
- Performance metrics
- Trained model artifacts
- Cross-validation scores

Run python -m mlflow ui in post 5000

## Containerization

### Build Docker Image

```bash
docker build -t heart-disease-api:latest .
```

### Run Container Locally

```bash
docker run -p 8000:8000 heart-disease-api:latest
```

Test: `curl http://localhost:8000/health`
Test: `curl http://localhost:8000/metrics`

## Kubernetes Deployment

### Prerequisites
- Docker Desktop with Kubernetes enabled OR Minikube installed

### Deploy to Kubernetes

```bash
# Load Docker image into Kubernetes
docker build -t heart-disease-api:latest .

# Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/grafana.yaml
kubectl apply -f k8s/prometheus.yaml

# Check deployment status
kubectl get pods
kubectl get svc

# Port forward to test locally
kubectl port-forward svc/heart-disease-api-service 8000:80
```

## CI/CD Pipeline

GitHub Actions workflow automatically:
1. Runs linting (flake8, black)
2. Executes unit tests (pytest)
3. Builds Docker image
4. Reports coverage metrics

Trigger: Push to `main` or `develop` branch

## Testing

### Run Unit Tests

```bash
pytest tests/ -v
```

### With Coverage

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Monitoring & Logging

### API Logging
- Request logging built into FastAPI app
- Logs saved to console and files

### Health Check Endpoint
```
GET /health
```

### Metrics Endpoint
```
GET /metrics
```

## API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information.

### 2. Health Check
```
GET /health
```
Returns health status of model and preprocessor.

### 3. Predict
```
POST /predict
Content-Type: application/json

{
  "age": float,
  "sex": int,
  "cp": int,
  "trestbps": float,
  "chol": float,
  "fbs": int,
  "restecg": int,
  "thalach": float,
  "exang": int,
  "oldpeak": float,
  "slope": int,
  "ca": int,
  "thal": int
}
```

Response:
```json
{
  "prediction": 0 or 1,
  "probability": float,
  "risk_level": "Low/Medium/High",
  "timestamp": "ISO-8601 timestamp"
}
```

## Key Features

✅ **Data Science**
- Comprehensive EDA with visualizations
- Data preprocessing and feature scaling
- Multiple model training and evaluation

✅ **Experiment Tracking**
- MLflow integration for reproducibility
- Parameter and metric logging
- Model versioning

✅ **Automation**
- GitHub Actions CI/CD pipeline
- Automated testing and linting
- Docker containerization

✅ **Deployment**
- Kubernetes-ready deployment manifests
- Load balancer service exposure
- Health checks and resource limits

✅ **Monitoring**
- API request logging
- Health check endpoint
- Error handling and reporting
- docker pull grafana/grafana:latest
- Grafana: http://localhost:30030/d/f60ca858-2fbf-4625-9eb3-bd7db99da881/heart-disease-api-metrics?orgId=1&from=now-5m&to=now&timezone=browser
- Prometheus: http://localhost:30090/graph?g0.expr=heart_disease_api_requests_created&g0.tab=1&g0.stacked=0&g0.show_exemplars=0&g0.range_input=1h
- Prometheus targets: http://localhost:30090/targets?search=

## CI CD
- Git Actions: act -j build-docker

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| ML Framework | Scikit-learn, XGBoost, SVM, RandomForest |
| API Framework | FastAPI |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Code Quality | Black, Flake8 |
| Monitoring | Grafana, Prometheus |

## Performance

### Model Metrics (Test Set)
| ============================================================
| FINAL EVALUATION - ALL MODELS
| ============================================================
|
| Baseline Models:
|
| LogisticRegression_Baseline Test Metrics:
|   accuracy_test: 0.8333
|   precision_test: 0.8462
|   recall_test: 0.7857
|   f1_test: 0.8148
|   roc_auc_test: 0.9498
|
| RandomForest_Baseline Test Metrics:
|   accuracy_test: 0.8667
|   precision_test: 0.8846
|   recall_test: 0.8214
|   f1_test: 0.8519
|   roc_auc_test: 0.9347
|
| SVM_Baseline Test Metrics:
|   accuracy_test: 0.8500
|   precision_test: 0.8800
|   recall_test: 0.7857
|   f1_test: 0.8302
|   roc_auc_test: 0.9554
|
| XGBoost_Baseline Test Metrics:
|   accuracy_test: 0.8667
|   precision_test: 0.8846
|   recall_test: 0.8214
|   f1_test: 0.8519
|   roc_auc_test: 0.8917
| Preprocessor saved to models/preprocessor.pkl
| Feature names saved to models/preprocessor.features.json
|
| ============================================================
| Training pipeline completed successfully!
| Best model: LogisticRegression_Grid


### API Response Time
- Average: <100ms
- Cold start: <1s

## Reproducibility

All components are designed for reproducibility:
1. `requirements.txt` ensures consistent dependencies
2. `preprocessor.pkl` ensures identical data transformations
3. `best_model.pkl` ensures consistent predictions
4. MLflow logs all experiment parameters
5. Docker container runs identically everywhere


## Authors

- Student ID: 2024ac05599@wilp.bits-pilani.ac.in

## License

MIT

## References

- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- MLflow Documentation: https://mlflow.org/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Kubernetes Documentation: https://kubernetes.io/

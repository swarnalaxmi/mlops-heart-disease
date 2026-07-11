# MLOps Assignment - Setup Instructions

## Prerequisites

- Python 3.11+
- Docker
- Git
- Kubernetes (Minikube or Docker Desktop Kubernetes)
- MLflow

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mlops-heart-disease.git
cd mlops-heart-disease
```

### 2. Set Up Python Environment

#### Option A: Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Option B: Using Conda
```bash
conda create -n mlops-heart python=3.11
conda activate mlops-heart
pip install -r requirements.txt
```

### 3. Download and Prepare Dataset

```bash
python src/model_trainer.py
```

This will automatically:
- Download the Heart Disease UCI dataset
- Handle missing values
- Preprocess and scale features
- Train two ML models
- Log experiments to MLflow

### 4. View MLflow Dashboard

```bash
mlflow ui
```
Then open: http://localhost:5000

## Running the API

### Option 1: Local Development

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

Access Swagger UI: http://localhost:8000/docs

### Option 2: Docker Container

```bash
# Build image
docker build -t heart-disease-api:latest .

# Run container
docker run -p 8000:8000 heart-disease-api:latest

# Test
curl http://localhost:8000/health
```

### Option 3: Kubernetes

```bash
# Build and load image
docker build -t heart-disease-api:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods
kubectl get svc

# Port forward for testing
kubectl port-forward svc/heart-disease-api-service 8000:80
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src

# Generate HTML coverage report
pytest tests/ -v --cov=src --cov-report=html
```

## Making Predictions

### Via cURL

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

### Via Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
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
}

response = requests.post(url, json=data)
print(response.json())
```

### Via Swagger UI

1. Go to http://localhost:8000/docs
2. Click "POST /predict"
3. Click "Try it out"
4. Enter sample values in the JSON body
5. Click "Execute"

## CI/CD Pipeline

The GitHub Actions workflow automatically runs on every push:

1. **Lint**: Code quality checks with flake8 and black
2. **Test**: Unit tests with pytest
3. **Build**: Docker image build validation

View workflow runs in GitHub repository under "Actions" tab.

## Project Structure Explanation

```
src/
  ├── app.py              # FastAPI application with /predict endpoint
  ├── data_processor.py   # Data loading, cleaning, preprocessing
  └── model_trainer.py    # Model training and MLflow logging

tests/
  ├── test_app.py         # FastAPI endpoint tests
  └── test_data_processor.py  # Data processor unit tests

k8s/
  ├── deployment.yaml     # Kubernetes deployment definition
  └── service.yaml        # Kubernetes service definition

.github/workflows/
  └── pipeline.yml        # GitHub Actions CI/CD pipeline

models/
  ├── best_model.pkl      # Trained Random Forest model
  └── preprocessor.pkl    # Data preprocessing pipeline

data/
  └── heart_disease.csv   # Downloaded dataset (auto-downloaded)
```

## Troubleshooting

### Model Not Loading
```bash
# Re-train the model
python src/model_trainer.py

# Check if files exist
ls models/
```

### Port Already in Use
```bash
# Free port 8000
# On Linux/Mac:
lsof -i :8000
kill -9 <PID>

# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Docker Build Fails
```bash
# Clean and rebuild
docker system prune -a
docker build --no-cache -t heart-disease-api:latest .
```

### Kubernetes Pod CrashLoop
```bash
# Check logs
kubectl logs <pod-name>

# Describe pod for details
kubectl describe pod <pod-name>

# Check if image is loaded
docker images | grep heart-disease-api
```

## Performance Metrics

Expected performance on test set:
- **Accuracy**: ~86%
- **Precision**: ~84%
- **Recall**: ~88%
- **ROC-AUC**: ~0.92

## Next Steps

1. Explore MLflow dashboard at http://localhost:5000
2. Test API with different patient profiles
3. Monitor API logs and metrics
4. Deploy to cloud if needed (AWS/GCP/Azure)
5. Implement model monitoring and retraining

## Support

For issues or questions, refer to:
- Project README: See README.md
- MLflow Docs: https://mlflow.org/docs/latest/
- FastAPI Docs: https://fastapi.tiangolo.com/
- Kubernetes Docs: https://kubernetes.io/docs/


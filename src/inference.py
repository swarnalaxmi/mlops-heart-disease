"""Inference module for making predictions with trained models."""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class ModelPredictor:
    """Load and use trained models for inference."""

    def __init__(self, model_path: str, preprocessor_path: str):
        """Initialize predictor with paths to model and preprocessor."""
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.model = None
        self.preprocessor = None
        self.load_models()

    def load_models(self):
        """Load model and preprocessor."""
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        if self.preprocessor_path.exists():
            self.preprocessor = joblib.load(self.preprocessor_path)
            print(f"Preprocessor loaded from {self.preprocessor_path}")
        else:
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")

    def preprocess_input(self, features: np.ndarray) -> np.ndarray:
        """Preprocess input features."""
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not loaded")
        return self.preprocessor.transform(features)

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction on input features.
        
        Args:
            features: Input features array
            
        Returns:
            Tuple of (prediction, probability)
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor not loaded")

        # Preprocess
        features_processed = self.preprocess_input(features)

        # Predict
        prediction = self.model.predict(features_processed)[0]
        probability = self.model.predict_proba(features_processed)[0][1]

        return int(prediction), float(probability)

    def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on batch of features.
        
        Args:
            features: Batch of input features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor not loaded")

        features_processed = self.preprocess_input(features)
        predictions = self.model.predict(features_processed)
        probabilities = self.model.predict_proba(features_processed)[:, 1]

        return predictions, probabilities


if __name__ == "__main__":
    # Test inference
    predictor = ModelPredictor("models/best_model.pkl", "models/preprocessor.pkl")

    # Sample input
    sample_features = np.array([[45, 1, 0, 120, 200, 0, 0, 100, 0, 0.5, 1, 0, 1]])

    prediction, probability = predictor.predict(sample_features)
    print(f"Prediction: {prediction}")
    print(f"Probability: {probability:.4f}")

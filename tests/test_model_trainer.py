"""Unit tests for model trainer."""

import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_trainer import ModelTrainer


def test_train_model_on_synthetic_data():
    """Test that ModelTrainer can train a model on synthetic data."""
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )

    trainer = ModelTrainer(experiment_name="test-heart-disease")
    model, metrics = trainer.train_model(
        "LogisticRegression",
        LogisticRegression(max_iter=1000, random_state=42),
        X,
        y
    )

    assert model is not None
    assert isinstance(metrics, dict)
    assert "accuracy_train" in metrics
    assert "roc_auc_train" in metrics
    assert metrics["accuracy_train"] >= 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

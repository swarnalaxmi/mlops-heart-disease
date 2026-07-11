"""pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope="session")
def sample_input_data():
    """Provide sample input data for tests."""
    return {
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


@pytest.fixture(scope="session")
def project_paths():
    """Provide project paths for tests."""
    project_root = Path(__file__).parent.parent
    return {
        "root": project_root,
        "data": project_root / "data",
        "models": project_root / "models",
        "src": project_root / "src",
        "tests": project_root / "tests"
    }

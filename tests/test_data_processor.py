"""Unit tests for data processor."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processor import DataProcessor


class TestDataProcessor:
    """Test DataProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DataProcessor()

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe."""
        data = {
            'age': [45, 50, 55, 60],
            'sex': [1, 0, 1, 0],
            'cp': [0, 1, 2, 3],
            'trestbps': [120.0, 130.0, 140.0, 150.0],
            'chol': [200.0, 210.0, 220.0, 230.0],
            'fbs': [0, 1, 0, 1],
            'restecg': [0, 1, 0, 1],
            'thalach': [100.0, 110.0, 120.0, 130.0],
            'exang': [0, 0, 1, 1],
            'oldpeak': [0.5, 1.0, 1.5, 2.0],
            'slope': [1, 1, 2, 2],
            'ca': [0, 0, 1, 1],
            'thal': [1, 2, 3, 1],
            'target': [0, 0, 1, 1]
        }
        return pd.DataFrame(data)

    def test_handle_missing_values(self, processor, sample_df):
        """Test missing value handling."""
        df_with_nan = sample_df.copy()
        df_with_nan.loc[0, 'age'] = np.nan
        
        result = processor.handle_missing_values(df_with_nan)
        
        assert result.isnull().sum().sum() == 0
        assert len(result) == 3

    def test_preprocess_data(self, processor, sample_df):
        """Test data preprocessing."""
        X, y = processor.preprocess_data(sample_df, fit=True)
        
        assert X is not None
        assert y is not None
        assert len(X) == len(sample_df)
        assert len(y) == len(sample_df)

    def test_feature_columns_set(self, processor, sample_df):
        """Test that feature columns are set after preprocessing."""
        X, y = processor.preprocess_data(sample_df, fit=True)
        
        assert processor.feature_columns is not None
        assert len(processor.feature_columns) == 13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

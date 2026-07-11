"""Data processing and preprocessing module."""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import json
from pathlib import Path
from ucimlrepo import fetch_ucirepo


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing."""

    def __init__(self, binary_target=True):
        self.preprocessor = None
        self.feature_columns = None
        self.target_column = 'target'
        self.binary_target = binary_target

    def load_dataset(self, filepath):
        """Load dataset from CSV."""
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape}")
        return df

    def handle_missing_values(self, df):
        """Handle missing values."""
        print(f"Missing values before:\n{df.isnull().sum()}")
        df = df.dropna()
        print(f"Missing values after: {df.isnull().sum().sum()}")
        return df

    def _binarize_target(self, y):
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=self.target_column)

        if self.binary_target:
            y = y.astype(int).apply(lambda value: 1 if value > 0 else 0)
        else:
            y = y.astype(int)

        y.name = self.target_column
        return y

    def preprocess_data(self, df, fit=False):
        """Preprocess data: encoding, scaling."""
        # Ensure the expected target column exists
        if self.target_column not in df.columns:
            df = df.rename(columns={df.columns[-1]: self.target_column})

        # Separate features and target
        X = df.drop(self.target_column, axis=1)
        y = self._binarize_target(df[self.target_column])

        self.feature_columns = X.columns.tolist()

        # Define numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Create preprocessor pipeline
        if fit:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='passthrough'
            )
            X_processed = self.preprocessor.fit_transform(X)

            # Capture output feature names for later use in inference
            try:
                # sklearn >=1.0 supports get_feature_names_out on ColumnTransformer
                self.feature_names_out = self.preprocessor.get_feature_names_out(self.feature_columns).tolist()
            except Exception:
                # Fallback: generate generic feature names
                self.feature_names_out = [f'feature_{i}' for i in range(X_processed.shape[1])]
        else:
            X_processed = self.preprocessor.transform(X)

        return X_processed, y

    def download_dataset(self, output_path='data/heart_disease.csv'):
        """Download Heart Disease UCI Dataset using ucimlrepo."""
        try:
            # Fetch the heart disease dataset from UCI ML Repository (ID: 45)
            heart_disease = fetch_ucirepo(id=45)

            # Extract features and targets
            X = heart_disease.data.features
            y = heart_disease.data.targets

            # Normalize feature input to a DataFrame
            if isinstance(X, pd.Series):
                X = X.to_frame()
            elif not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            # Normalize target input to a Series
            if isinstance(y, pd.DataFrame):
                if y.shape[1] == 1:
                    y = y.iloc[:, 0]
                else:
                    y = y.iloc[:, 0]
            elif not isinstance(y, pd.Series):
                y = pd.Series(y)

            y = y.rename(self.target_column)

            # Combine features and targets into a single dataframe
            df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

            # If the combined dataframe still does not have the expected target column,
            # rename the final column to the configured target name.
            if self.target_column not in df.columns:
                df.columns = list(X.columns) + [self.target_column]

            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            df.to_csv(output_path, index=False)
            print(f"Dataset downloaded and saved to {output_path}")
            print(f"Dataset shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None

    def save_preprocessor(self, filepath):
        """Save preprocessor pipeline."""
        joblib.dump(self.preprocessor, filepath)
        # Save feature names alongside the preprocessor for inference
        features_path = Path(filepath).with_suffix('.features.json')
        try:
            with open(features_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_names_out if hasattr(self, 'feature_names_out') else self.feature_columns, f)
        except Exception:
            pass

        print(f"Preprocessor saved to {filepath}")
        print(f"Feature names saved to {features_path}")

    def load_preprocessor(self, filepath):
        """Load preprocessor pipeline."""
        self.preprocessor = joblib.load(filepath)
        # Try to load associated feature names
        features_path = Path(filepath).with_suffix('.features.json')
        if features_path.exists():
            try:
                with open(features_path, 'r', encoding='utf-8') as f:
                    self.feature_names_out = json.load(f)
            except Exception:
                self.feature_names_out = None
        else:
            self.feature_names_out = None

        print(f"Preprocessor loaded from {filepath}")
        if self.feature_names_out is not None:
            print(f"Loaded {len(self.feature_names_out)} feature names")


if __name__ == "__main__":
    processor = DataProcessor()
    df = processor.download_dataset()
    if df is not None:
        print(f"\nDataset Info:\n{df.info()}")
        print(f"\nFirst few rows:\n{df.head()}")

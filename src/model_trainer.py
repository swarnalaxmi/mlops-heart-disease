"""Model training and evaluation module."""

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, make_scorer, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
import numpy as np
from data_processor import DataProcessor


class ModelTrainer:
    """Trains and evaluates ML models."""

    def __init__(self, experiment_name="heart-disease-prediction"):
        self.models = {}
        self.best_model = None
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def _get_classification_average(self, y):
        unique_labels = np.unique(y)
        return 'binary' if len(unique_labels) == 2 else 'weighted'

    def _get_roc_auc(self, y_true, y_proba):
        unique_labels = np.unique(y_true)
        if y_proba.ndim == 2 and y_proba.shape[1] == 2 and len(unique_labels) == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        if y_proba.ndim == 1 or y_proba.shape[1] == 1:
            return roc_auc_score(y_true, y_proba)
        return roc_auc_score(y_true, y_proba, multi_class='ovo', average='weighted')

    def _get_multiclass_scoring(self, y):
        unique_labels = np.unique(y)
        is_binary = len(unique_labels) == 2
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(
                precision_score,
                average='binary' if is_binary else 'weighted',
                zero_division=0
            ),
            'recall': make_scorer(
                recall_score,
                average='binary' if is_binary else 'weighted',
                zero_division=0
            ),
            'f1': make_scorer(
                f1_score,
                average='binary' if is_binary else 'weighted',
                zero_division=0
            )
        }
        if is_binary:
            scorers['roc_auc'] = make_scorer(roc_auc_score, needs_proba=True)
        else:
            scorers['roc_auc'] = make_scorer(
                roc_auc_score,
                needs_proba=True,
                multi_class='ovo',
                average='weighted'
            )
        return scorers

    def _get_search_scoring(self, y):
        unique_labels = np.unique(y)
        return 'roc_auc' if len(unique_labels) == 2 else 'roc_auc_ovo_weighted'

    def train_model(self, model_name, model, X_train, y_train):
        """Train a single model."""
        print(f"\nTraining {model_name}...")
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            self.models[model_name] = model

            # Get model parameters
            params = model.get_params()
            mlflow.log_params(params)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_proba = model.predict_proba(X_train)
            average_method = self._get_classification_average(y_train)

            # Calculate metrics
            metrics = {
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'precision_train': precision_score(y_train, y_pred_train, average=average_method),
                'recall_train': recall_score(y_train, y_pred_train, average=average_method),
                'f1_train': f1_score(y_train, y_pred_train, average=average_method),
                'roc_auc_train': self._get_roc_auc(y_train, y_pred_proba)
            }

            mlflow.log_metrics(metrics)
            
            # Cross-validation scores
            print(f"\nPerforming 5-fold cross-validation for {model_name}...")
            scoring = self._get_multiclass_scoring(y_train)
            
            cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring, return_train_score=True)
            
            # Log cross-validation scores
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                cv_mean = np.mean(cv_results[f'test_{metric}'])
                cv_std = np.std(cv_results[f'test_{metric}'])
                mlflow.log_metric(f'cv_{metric}_mean', cv_mean)
                mlflow.log_metric(f'cv_{metric}_std', cv_std)
                print(f"  {metric}: {cv_mean:.4f} (+/- {cv_std:.4f})")

            # Log training plots
            self._log_roc_curve(model_name, y_train, y_pred_proba)
            self._log_confusion_matrix(model_name, y_train, y_pred_train)
            
            mlflow.sklearn.log_model(model, f"models/{model_name}")

            print(f"{model_name} training metrics: {metrics}")
            return model, metrics

    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate model on test set."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        average_method = self._get_classification_average(y_test)

        metrics = {
            'accuracy_test': accuracy_score(y_test, y_pred),
            'precision_test': precision_score(y_test, y_pred, average=average_method),
            'recall_test': recall_score(y_test, y_pred, average=average_method),
            'f1_test': f1_score(y_test, y_pred, average=average_method),
            'roc_auc_test': self._get_roc_auc(y_test, y_pred_proba)
        }

        print(f"\n{model_name} Test Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
            mlflow.log_metric(key, value)

        # Log evaluation plots to MLflow
        self._log_roc_curve(model_name, y_test, y_pred_proba)
        self._log_confusion_matrix(model_name, y_test, y_pred)

        return metrics

    def _log_plot(self, run, figure, filename):
        """Save and log a matplotlib figure to MLflow."""
        output_dir = Path('mlflow_artifacts')
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / filename
        figure.savefig(plot_path, bbox_inches='tight')
        plt.close(figure)
        mlflow.log_artifact(str(plot_path))

    def _log_roc_curve(self, model_name, y_true, y_proba):
        """Create and log ROC curve plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = np.unique(y_true)

        if len(unique_labels) == 2:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
        else:
            y_true_binarized = label_binarize(y_true, classes=unique_labels)
            for idx, label in enumerate(unique_labels):
                if y_proba.ndim == 1:
                    continue
                RocCurveDisplay.from_predictions(
                    y_true_binarized[:, idx],
                    y_proba[:, idx],
                    name=f'class {label}',
                    ax=ax
                )
            ax.legend(loc='lower right')

        ax.set_title(f'{model_name} ROC Curve')
        self._log_plot(mlflow.active_run(), fig, f'{model_name}_roc_curve.png')

    def _log_confusion_matrix(self, model_name, y_true, y_pred):
        """Create and log confusion matrix plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
        ax.set_title(f'{model_name} Confusion Matrix')
        self._log_plot(mlflow.active_run(), fig, f'{model_name}_confusion_matrix.png')

    def _ensure_no_active_run(self):
        if mlflow.active_run() is not None:
            mlflow.end_run()

    def hyperparameter_tuning_grid(self, model_name, X_train, y_train):
        """Perform GridSearchCV for hyperparameter tuning."""
        print(f"\n{'='*50}")
        print(f"GridSearchCV Hyperparameter Tuning: {model_name}")
        print(f"{'='*50}")
        
        self._ensure_no_active_run()
        with mlflow.start_run(run_name=f"{model_name}_GridSearch", nested=True):
            if model_name == "LogisticRegression":
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [500, 1000, 2000]
                }
                base_model = LogisticRegression(random_state=42, class_weight='balanced')
            elif model_name == "RandomForest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
                base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            elif model_name == "SVM":
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
                base_model = SVC(probability=True, random_state=42)
            elif model_name == "XGBoost":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 1.0]
                }
                base_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1)
            else:
                return None

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring=self._get_search_scoring(y_train),
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)

            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best CV ROC-AUC score: {grid_search.best_score_:.4f}")

            # Log best parameters
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric('grid_search_best_roc_auc', grid_search.best_score_)
            mlflow.sklearn.log_model(grid_search.best_estimator_, f"models/{model_name}_GridSearch")

            return grid_search.best_estimator_, grid_search.best_params_

    def hyperparameter_tuning_random(self, model_name, X_train, y_train):
        """Perform RandomizedSearchCV for hyperparameter tuning."""
        print(f"\n{'='*50}")
        print(f"RandomizedSearchCV Hyperparameter Tuning: {model_name}")
        print(f"{'='*50}")
        
        self._ensure_no_active_run()
        with mlflow.start_run(run_name=f"{model_name}_RandomSearch", nested=True):
            if model_name == "LogisticRegression":
                param_dist = {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [100, 500, 1000]
                }
                base_model = LogisticRegression(random_state=42, class_weight='balanced')
            elif model_name == "RandomForest":
                param_dist = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
                base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
            elif model_name == "SVM":
                param_dist = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
                base_model = SVC(probability=True, random_state=42)
            elif model_name == "XGBoost":
                param_dist = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 1.0]
                }
                base_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1)
            else:
                return None

            random_search = RandomizedSearchCV(
                base_model,
                param_dist,
                n_iter=20,
                cv=5,
                scoring=self._get_search_scoring(y_train),
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            random_search.fit(X_train, y_train)

            print(f"\nBest parameters: {random_search.best_params_}")
            print(f"Best CV ROC-AUC score: {random_search.best_score_:.4f}")

            # Log best parameters
            mlflow.log_params(random_search.best_params_)
            mlflow.log_metric('random_search_best_roc_auc', random_search.best_score_)
            mlflow.sklearn.log_model(random_search.best_estimator_, f"models/{model_name}_RandomSearch")

            return random_search.best_estimator_, random_search.best_params_

    def save_model(self, model_name, filepath):
        """Save trained model."""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            print(f"Model {model_name} saved to {filepath}")
        else:
            print(f"Model {model_name} not found in trained models")

    def load_model(self, filepath):
        """Load a trained model."""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model


def run_training_pipeline(data_path='data/heart_disease.csv', output_model_path='models/best_model.pkl'):
    """Complete training pipeline."""
    
    # Download and load data
    processor = DataProcessor()
    df = processor.download_dataset(data_path)
    
    if df is None:
        print("Failed to download dataset")
        return None, None

    # Handle missing values
    df = processor.handle_missing_values(df)

    # Preprocess data
    X, y = processor.preprocess_data(df, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Initialize trainer
    trainer = ModelTrainer()

    # ===== BASELINE MODELS =====
    print("\n" + "="*60)
    print("TRAINING BASELINE MODELS")
    print("="*60)

    lr_model, lr_metrics_train = trainer.train_model(
        "LogisticRegression",
        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        X_train,
        y_train
    )

    rf_model, rf_metrics_train = trainer.train_model(
        "RandomForest",
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        X_train,
        y_train
    )

    svm_model, svm_metrics_train = trainer.train_model(
        "SVM",
        SVC(probability=True, random_state=42),
        X_train,
        y_train
    )

    xgb_model, xgb_metrics_train = trainer.train_model(
        "XGBoost",
        XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1),
        X_train,
        y_train
    )

    # ===== HYPERPARAMETER TUNING =====
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)

    tuned_models = {}
    for model_name in ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]:
        grid_model, grid_params = trainer.hyperparameter_tuning_grid(model_name, X_train, y_train)
        if grid_model is not None:
            tuned_models[f"{model_name}_Grid"] = (
                grid_model,
                trainer.evaluate_model(f"{model_name}_Grid_Tuned", grid_model, X_test, y_test),
                grid_params
            )

        random_model, random_params = trainer.hyperparameter_tuning_random(model_name, X_train, y_train)
        if random_model is not None:
            tuned_models[f"{model_name}_Random"] = (
                random_model,
                trainer.evaluate_model(f"{model_name}_Random_Tuned", random_model, X_test, y_test),
                random_params
            )

    # ===== FINAL EVALUATION =====
    print("\n" + "="*60)
    print("FINAL EVALUATION - ALL MODELS")
    print("="*60)

    baseline_models = {
        "LogisticRegression_Baseline": lr_model,
        "RandomForest_Baseline": rf_model,
        "SVM_Baseline": svm_model,
        "XGBoost_Baseline": xgb_model
    }

    baseline_metrics = {}
    print("\nBaseline Models:")
    for name, model in baseline_models.items():
        baseline_metrics[name] = trainer.evaluate_model(name, model, X_test, y_test)

    if tuned_models:
        best_key, (best_model, best_metrics, best_params) = max(
            tuned_models.items(),
            key=lambda item: item[1][1].get('roc_auc_test', -np.inf)
        )
        best_name = best_key
    else:
        best_key, best_metrics = max(
            baseline_metrics.items(),
            key=lambda item: item[1].get('roc_auc_test', -np.inf)
        )
        best_model = baseline_models[best_key]
        best_name = best_key

    # Save best model and preprocessor
    joblib.dump(best_model, output_model_path)
    processor.save_preprocessor('models/preprocessor.pkl')

    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print(f"Best model: {best_name}")
    print("="*60)

    return trainer, processor


if __name__ == "__main__":
    trainer, processor = run_training_pipeline()

"""Model training module for bank marketing prediction."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.utils.data_loader import DataLoader
from src.utils.preprocessing import DataPreprocessor


class ModelTrainer:
    """Train and evaluate prediction models for bank marketing."""

    def __init__(self, model_dir: Path | None = None):
        """Initialize the model trainer.

        Args:
            model_dir: Directory to save trained models. Defaults to /models from project root.
        """
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent.parent / "models"
        else:
            self.model_dir = Path(model_dir)

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Define models to try
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
        }

    def load_and_prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load training data and split into train/validation sets.

        Returns:
            Tuple of (train_df, val_df).
        """
        loader = DataLoader()
        df = loader.load_train()

        # Drop id column as it's not useful for prediction
        df = df.drop(columns=["id"])

        # Split data
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["subscribe"]
        )

        return train_df, val_df

    def train_model(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        preprocessor: DataPreprocessor,
    ) -> tuple:
        """Train a single model.

        Args:
            model_name: Name of the model to train.
            train_df: Training DataFrame.
            preprocessor: Fitted preprocessor.

        Returns:
            Tuple of (model, training_time_seconds).
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        model = self.models[model_name]

        # Prepare features and target
        features, target = preprocessor.fit_transform(train_df)

        # Train model
        model.fit(features, target)

        return model

    def evaluate_model(
        self,
        model,
        val_df: pd.DataFrame,
        preprocessor: DataPreprocessor,
    ) -> dict:
        """Evaluate model performance on validation data.

        Args:
            model: Trained model.
            val_df: Validation DataFrame.
            preprocessor: Fitted preprocessor.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Prepare features and target
        features = preprocessor.transform(val_df)
        target = preprocessor.transform_target(val_df["subscribe"])

        # Get predictions
        y_pred = model.predict(features)
        y_proba = model.predict_proba(features)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(target, y_pred),
            "precision": precision_score(target, y_pred, average="binary"),
            "recall": recall_score(target, y_pred, average="binary"),
            "f1": f1_score(target, y_pred, average="binary"),
            "auc_roc": roc_auc_score(target, y_proba),
        }

        return metrics

    def train_all_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> dict[str, dict]:
        """Train all models and evaluate them.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.

        Returns:
            Dictionary mapping model names to their metrics.
        """
        results = {}
        preprocessor = DataPreprocessor()
        preprocessor.fit(train_df)

        for model_name in self.models.keys():
            # Train model
            model = self.train_model(model_name, train_df, preprocessor)

            # Evaluate
            metrics = self.evaluate_model(model, val_df, preprocessor)

            # Store results
            results[model_name] = {
                "metrics": metrics,
                "model": model,
                "preprocessor": preprocessor,
            }

            print(f"{model_name}: AUC-ROC = {metrics['auc_roc']:.4f}, F1 = {metrics['f1']:.4f}")

        return results

    def select_best_model(self, results: dict[str, dict], metric: str = "auc_roc") -> tuple:
        """Select the best model based on a metric.

        Args:
            results: Results from train_all_models.
            metric: Metric to use for selection (default: auc_roc).

        Returns:
            Tuple of (best_model_name, best_model, best_preprocessor).
        """
        best_name = max(results.keys(), key=lambda k: results[k]["metrics"][metric])

        return (
            best_name,
            results[best_name]["model"],
            results[best_name]["preprocessor"],
        )

    def save_model(self, model, preprocessor: DataPreprocessor, model_name: str = "model") -> Path:
        """Save trained model and preprocessor to disk.

        Args:
            model: Trained model.
            preprocessor: Fitted preprocessor.
            model_name: Name for the saved model files.

        Returns:
            Path to the saved model file.
        """
        model_path = self.model_dir / f"{model_name}.pkl"
        preprocessor_path = self.model_dir / f"{model_name}_preprocessor.pkl"

        # Save model and preprocessor
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        print(f"Model saved to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")

        return model_path

    def save_metrics(self, metrics: dict, model_name: str = "model") -> Path:
        """Save evaluation metrics to disk.

        Args:
            metrics: Metrics dictionary.
            model_name: Name for the metrics file.

        Returns:
            Path to the saved metrics file.
        """
        metrics_path = self.model_dir / f"{model_name}_metrics.json"

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {metrics_path}")

        return metrics_path

    def train_and_save(self, model_name: str = "gradient_boosting") -> tuple[Path, Path]:
        """Full training pipeline: load data, train, evaluate, and save.

        Args:
            model_name: Name of the model to train.

        Returns:
            Tuple of (model_path, metrics_path).
        """
        print("Loading data...")
        train_df, val_df = self.load_and_prepare_data()
        print(f"Training set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")

        print("\nTraining models...")
        results = self.train_all_models(train_df, val_df)

        print("\nSelecting best model...")
        best_name, best_model, best_preprocessor = self.select_best_model(results)
        print(f"Best model: {best_name}")

        print("\nSaving best model...")
        model_path = self.save_model(best_model, best_preprocessor, model_name)

        print("\nSaving metrics...")
        metrics_path = self.save_metrics(results, model_name)

        # Print detailed classification report
        print("\nClassification Report:")
        predictions = best_model.predict(best_preprocessor.transform(val_df))
        target = best_preprocessor.transform_target(val_df["subscribe"])
        print(classification_report(target, predictions, target_names=["no", "yes"]))

        # Feature importance
        if hasattr(best_model, "feature_importances_"):
            print("\nTop 10 Feature Importances:")
            feature_names = best_preprocessor.get_feature_names()
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": best_model.feature_importances_}
            ).sort_values("importance", ascending=False)
            print(importance_df.head(10).to_string(index=False))

        return model_path, metrics_path


def main():
    """Main training function."""
    trainer = ModelTrainer()
    model_path, metrics_path = trainer.train_and_save()

    print(f"\n✅ Training complete!")
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()

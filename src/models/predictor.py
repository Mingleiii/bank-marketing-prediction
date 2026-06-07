"""Model prediction module for bank marketing prediction."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.data_loader import DataLoader
from src.utils.preprocessing import prepare_prediction_input


class ModelPredictor:
    """Load trained model and make predictions."""

    def __init__(self, model_path: Path | None = None, preprocessor_path: Path | None = None):
        """Initialize the predictor.

        Args:
            model_path: Path to the trained model file.
            preprocessor_path: Path to the preprocessor file.
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "model.pkl"
        else:
            model_path = Path(model_path)

        if preprocessor_path is None:
            preprocessor_path = (
                Path(__file__).parent.parent.parent / "models" / "model_preprocessor.pkl"
            )
        else:
            preprocessor_path = Path(preprocessor_path)

        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.categorical_values = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model and preprocessor from disk.

        Raises:
            FileNotFoundError: If model or preprocessor files don't exist.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        # Load categorical values from training data
        self._load_categorical_values()

    def _load_categorical_values(self) -> None:
        """Load valid categorical values from training data."""
        loader = DataLoader()
        train_df = loader.load_train()
        self.categorical_values = loader.get_categorical_values(train_df)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Make a prediction for a single sample.

        Args:
            data: Dictionary containing feature values.

        Returns:
            Dictionary with prediction results.
        """
        # Prepare input
        input_df = prepare_prediction_input(data, self.categorical_values)

        # Transform features
        features = self.preprocessor.transform(input_df)

        # Get prediction
        prediction_proba = self.model.predict_proba(features)[0]
        prediction_label = self.model.predict(features)[0]

        # Convert to original label
        original_label = self.preprocessor.inverse_transform_target([prediction_label])[0]
        subscribe_probability = prediction_proba[1]  # Probability of 'yes'

        return {
            "prediction": original_label,  # 'yes' or 'no'
            "subscribe_probability": float(subscribe_probability),
            "not_subscribe_probability": float(prediction_proba[0]),
            "confidence": float(max(prediction_proba)),
        }

    def predict_batch(self, data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Make predictions for multiple samples.

        Args:
            data_list: List of dictionaries containing feature values.

        Returns:
            List of prediction result dictionaries.
        """
        results = []
        for data in data_list:
            result = self.predict(data)
            results.append(result)
        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model.

        Returns:
            DataFrame with feature names and importance scores.

        Raises:
            ValueError: If model doesn't support feature importance.
        """
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not support feature importance")

        feature_names = self.preprocessor.get_feature_names()
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return importance_df

    def get_categorical_options(self) -> dict[str, list[str]]:
        """Get valid options for categorical fields.

        Returns:
            Dictionary mapping field names to their valid values.
        """
        return self.categorical_values

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        info = {
            "model_type": type(self.model).__name__,
            "model_path": str(self.model_path),
            "preprocessor_path": str(self.preprocessor_path),
            "is_fitted": True,
        }

        if hasattr(self.model, "feature_importances_"):
            info["supports_feature_importance"] = True
        else:
            info["supports_feature_importance"] = False

        if hasattr(self.model, "predict_proba"):
            info["supports_probability"] = True
        else:
            info["supports_probability"] = False

        return info

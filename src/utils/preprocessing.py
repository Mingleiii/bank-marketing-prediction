"""Data preprocessing module for bank marketing dataset."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class DataPreprocessor:
    """Preprocess bank marketing data for model training and prediction."""

    # Categorical columns that need encoding
    CATEGORICAL_COLUMNS = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
    ]

    # Numerical columns that need scaling
    NUMERICAL_COLUMNS = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp_var_rate",
        "cons_price_index",
        "cons_conf_index",
        "lending_rate3m",
        "nr_employed",
    ]

    def __init__(self):
        """Initialize the preprocessor."""
        self.feature_columns = self.CATEGORICAL_COLUMNS + self.NUMERICAL_COLUMNS
        self.column_transformer: ColumnTransformer | None = None
        self.target_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_column: str = "subscribe") -> "DataPreprocessor":
        """Fit the preprocessor on training data.

        Args:
            df: Training DataFrame.
            target_column: Name of the target column.

        Returns:
            Self for method chaining.
        """
        # Validate columns exist
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        # Fit target encoder
        self.target_encoder.fit(df[target_column])

        # Build column transformer
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        numerical_transformer = StandardScaler()

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, self.CATEGORICAL_COLUMNS),
                ("num", numerical_transformer, self.NUMERICAL_COLUMNS),
            ],
            remainder="drop",
        )

        # Fit on features
        self.column_transformer.fit(df[self.feature_columns])
        self.is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame using fitted preprocessor.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame with encoded features.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Transform features
        features = self.column_transformer.transform(df[self.feature_columns])

        # Get feature names
        cat_features = self.column_transformer.named_transformers_["cat"].get_feature_names_out(
            self.CATEGORICAL_COLUMNS
        )
        feature_names = list(cat_features) + self.NUMERICAL_COLUMNS

        # Create DataFrame
        result = pd.DataFrame(features, columns=feature_names, index=df.index)

        return result

    def fit_transform(
        self, df: pd.DataFrame, target_column: str = "subscribe"
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Fit and transform in one step.

        Args:
            df: DataFrame to process.
            target_column: Name of the target column.

        Returns:
            Tuple of (transformed_features, encoded_target).
        """
        self.fit(df, target_column)
        features = self.transform(df)
        target = self.transform_target(df[target_column])
        return features, target

    def transform_target(self, target: pd.Series) -> np.ndarray:
        """Transform target values using fitted encoder.

        Args:
            target: Target series to transform.

        Returns:
            Encoded target array (0/1).
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before target transform")
        return self.target_encoder.transform(target)

    def inverse_transform_target(self, target_encoded: np.ndarray) -> np.ndarray:
        """Inverse transform encoded target values.

        Args:
            target_encoded: Encoded target array (0/1).

        Returns:
            Original target values (e.g., 'yes'/'no').
        """
        return self.target_encoder.inverse_transform(target_encoded)

    def get_feature_names(self) -> list[str]:
        """Get names of all output features.

        Returns:
            List of feature names after transformation.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")

        cat_features = self.column_transformer.named_transformers_["cat"].get_feature_names_out(
            self.CATEGORICAL_COLUMNS
        )
        return list(cat_features) + self.NUMERICAL_COLUMNS


def prepare_prediction_input(
    data: dict[str, Any],
    categorical_values: dict[str, list[str]],
) -> pd.DataFrame:
    """Prepare a single prediction input as DataFrame.

    Args:
        data: Dictionary containing feature values.
        categorical_values: Dictionary of valid categorical values.

    Returns:
        DataFrame ready for transformation.

    Raises:
        ValueError: If required keys are missing or values are invalid.
    """
    required_keys = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "emp_var_rate",
        "cons_price_index",
        "cons_conf_index",
        "lending_rate3m",
        "nr_employed",
    ]

    missing_keys = set(required_keys) - set(data.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    # Validate categorical values
    for col in DataPreprocessor.CATEGORICAL_COLUMNS:
        if col in categorical_values and data[col] not in categorical_values[col]:
            raise ValueError(f"Invalid value for {col}: {data[col]}")

    # Convert to DataFrame
    df = pd.DataFrame([data])
    return df[required_keys]

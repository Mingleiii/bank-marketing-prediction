"""Data loading module for bank marketing dataset."""

from pathlib import Path
from typing import Optional

import pandas as pd


class DataLoader:
    """Load and validate bank marketing dataset."""

    # Required columns in the dataset
    REQUIRED_COLUMNS = [
        "id",
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
        "subscribe",
    ]

    # Categorical columns
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
        "subscribe",
    ]

    # Numerical columns
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

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the data loader.

        Args:
            data_dir: Path to the data directory. Defaults to ../data from src.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.train_path = self.data_dir / "train.csv"
        self.test_path = self.data_dir / "test.csv"

    def load_train(self) -> pd.DataFrame:
        """Load training data.

        Returns:
            DataFrame containing training data.

        Raises:
            FileNotFoundError: If train.csv does not exist.
            ValueError: If CSV format is invalid.
        """
        if not self.train_path.exists():
            raise FileNotFoundError(f"Training file not found: {self.train_path}")

        try:
            df = pd.read_csv(self.train_path)
        except Exception as e:
            raise ValueError(f"Failed to load training data: {e}") from e

        self._validate_columns(df)
        return df

    def load_test(self) -> pd.DataFrame:
        """Load test data.

        Returns:
            DataFrame containing test data.

        Raises:
            FileNotFoundError: If test.csv does not exist.
            ValueError: If CSV format is invalid.
        """
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test file not found: {self.test_path}")

        try:
            df = pd.read_csv(self.test_path)
        except Exception as e:
            raise ValueError(f"Failed to load test data: {e}") from e

        self._validate_columns(df)
        return df

    def load_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load both training and test data.

        Returns:
            Tuple of (train_df, test_df).
        """
        return self.load_train(), self.load_test()

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist in the DataFrame.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If required columns are missing.
        """
        missing_columns = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the dataset.

        Args:
            df: DataFrame to summarize.

        Returns:
            Dictionary containing summary statistics.
        """
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        # Target distribution
        if "subscribe" in df.columns:
            value_counts = df["subscribe"].value_counts()
            summary["subscribe_distribution"] = {
                "yes": int(value_counts.get("yes", 0)),
                "no": int(value_counts.get("no", 0)),
                "yes_rate": float(value_counts.get("yes", 0) / len(df)),
            }

        # Numerical statistics
        numerical_stats = {}
        for col in self.NUMERICAL_COLUMNS:
            if col in df.columns:
                col_stats = df[col].describe()
                numerical_stats[col] = {
                    "mean": float(col_stats["mean"]),
                    "std": float(col_stats["std"]),
                    "min": float(col_stats["min"]),
                    "max": float(col_stats["max"]),
                }
        summary["numerical_stats"] = numerical_stats

        return summary

    def get_categorical_values(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Get unique values for categorical columns.

        Args:
            df: DataFrame to analyze.

        Returns:
            Dictionary mapping column names to their unique values.
        """
        result = {}
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                result[col] = sorted(df[col].dropna().unique().tolist())
        return result

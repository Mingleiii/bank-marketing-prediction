"""Tests for data loader module."""

import pandas as pd
import pytest
from src.utils.data_loader import DataLoader


@pytest.fixture
def data_loader():
    """Create a data loader instance."""
    return DataLoader()


@pytest.fixture
def sample_df():
    """Create a sample DataFrame matching bank data structure."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "age": [30, 45],
            "job": ["admin.", "services"],
            "marital": ["married", "single"],
            "education": ["university.degree", "high.school"],
            "default": ["no", "no"],
            "housing": ["yes", "no"],
            "loan": ["no", "yes"],
            "contact": ["cellular", "telephone"],
            "month": ["may", "aug"],
            "day_of_week": ["mon", "tue"],
            "duration": [300, 500],
            "campaign": [1, 2],
            "pdays": [999, 10],
            "previous": [0, 1],
            "poutcome": ["nonexistent", "success"],
            "emp_var_rate": [1.1, -0.1],
            "cons_price_index": [93.994, 94.000],
            "cons_conf_index": [-36.4, -40.0],
            "lending_rate3m": [4.85, 4.9],
            "nr_employed": [5191.0, 5200.0],
            "subscribe": ["no", "yes"],
        }
    )


def test_required_columns_exist():
    """Test that required columns are defined."""
    assert len(DataLoader.REQUIRED_COLUMNS) == 22
    assert "subscribe" in DataLoader.REQUIRED_COLUMNS
    assert "age" in DataLoader.REQUIRED_COLUMNS


def test_categorical_columns_exist():
    """Test that categorical columns are defined."""
    assert len(DataLoader.CATEGORICAL_COLUMNS) == 11
    assert "subscribe" in DataLoader.CATEGORICAL_COLUMNS
    assert "job" in DataLoader.CATEGORICAL_COLUMNS


def test_numerical_columns_exist():
    """Test that numerical columns are defined."""
    assert len(DataLoader.NUMERICAL_COLUMNS) == 10
    assert "age" in DataLoader.NUMERICAL_COLUMNS
    assert "subscribe" not in DataLoader.NUMERICAL_COLUMNS


def test_data_loader_initialization(data_loader):
    """Test data loader initialization."""
    assert data_loader.data_dir.name == "data"
    assert data_loader.train_path.name == "train.csv"
    assert data_loader.test_path.name == "test.csv"


def test_validate_columns_valid(data_loader, sample_df):
    """Test column validation with valid DataFrame."""
    # Should not raise
    data_loader._validate_columns(sample_df)


def test_validate_columns_missing(data_loader):
    """Test column validation with missing columns."""
    invalid_df = pd.DataFrame({"age": [30, 45], "job": ["admin.", "services"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        data_loader._validate_columns(invalid_df)


def test_get_data_summary(data_loader, sample_df):
    """Test data summary generation."""
    summary = data_loader.get_data_summary(sample_df)

    assert summary["total_rows"] == 2
    assert summary["total_columns"] == 22
    assert "subscribe_distribution" in summary
    assert summary["subscribe_distribution"]["yes"] == 1
    assert summary["subscribe_distribution"]["no"] == 1
    assert "numerical_stats" in summary


def test_get_categorical_values(data_loader, sample_df):
    """Test categorical values extraction."""
    cat_values = data_loader.get_categorical_values(sample_df)

    assert "job" in cat_values
    assert sorted(cat_values["job"]) == ["admin.", "services"]
    assert "subscribe" in cat_values
    assert "no" in cat_values["subscribe"]
    assert "yes" in cat_values["subscribe"]


def test_train_file_not_found(tmp_path):
    """Test error when training file doesn't exist."""
    loader = DataLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_train()


def test_test_file_not_found(tmp_path):
    """Test error when test file doesn't exist."""
    # Create train file
    (tmp_path / "train.csv").write_text("age,subscribe\n30,no\n")

    loader = DataLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_test()

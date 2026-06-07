"""Tests for data preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from src.utils.preprocessing import DataPreprocessor, prepare_prediction_input


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for preprocessing."""
    return pd.DataFrame(
        {
            "age": [30, 45, 25],
            "job": ["admin.", "services", "admin."],
            "marital": ["married", "single", "single"],
            "education": ["university.degree", "high.school", "high.school"],
            "default": ["no", "no", "no"],
            "housing": ["yes", "no", "yes"],
            "loan": ["no", "yes", "no"],
            "contact": ["cellular", "telephone", "cellular"],
            "month": ["may", "aug", "may"],
            "day_of_week": ["mon", "tue", "mon"],
            "duration": [300, 500, 200],
            "campaign": [1, 2, 1],
            "pdays": [999, 10, 999],
            "previous": [0, 1, 0],
            "poutcome": ["nonexistent", "success", "nonexistent"],
            "emp_var_rate": [1.1, -0.1, 1.1],
            "cons_price_index": [93.994, 94.000, 93.994],
            "cons_conf_index": [-36.4, -40.0, -36.4],
            "lending_rate3m": [4.85, 4.9, 4.85],
            "nr_employed": [5191.0, 5200.0, 5191.0],
            "subscribe": ["no", "yes", "no"],
        }
    )


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance."""
    return DataPreprocessor()


def test_preprocessor_initialization(preprocessor):
    """Test preprocessor initialization."""
    assert len(preprocessor.CATEGORICAL_COLUMNS) == 10
    assert len(preprocessor.NUMERICAL_COLUMNS) == 10
    assert not preprocessor.is_fitted


def test_preprocessor_fit(preprocessor, sample_df):
    """Test fitting the preprocessor."""
    preprocessor.fit(sample_df)
    assert preprocessor.is_fitted
    assert preprocessor.column_transformer is not None


def test_preprocessor_fit_missing_column(preprocessor, sample_df):
    """Test fitting with missing columns."""
    incomplete_df = sample_df.drop(columns=["age"])
    with pytest.raises(ValueError, match="Missing feature columns"):
        preprocessor.fit(incomplete_df)


def test_preprocessor_fit_missing_target(preprocessor, sample_df):
    """Test fitting with missing target column."""
    no_target_df = sample_df.drop(columns=["subscribe"])
    with pytest.raises(ValueError, match="Target column"):
        preprocessor.fit(no_target_df)


def test_preprocessor_transform_before_fit(preprocessor, sample_df):
    """Test transforming before fitting raises error."""
    with pytest.raises(ValueError, match="must be fitted"):
        preprocessor.transform(sample_df)


def test_preprocessor_fit_transform(preprocessor, sample_df):
    """Test fit and transform."""
    features, target = preprocessor.fit_transform(sample_df)

    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == 3
    assert len(target) == 3
    assert set(target) == {0, 1}


def test_preprocessor_transform_target(preprocessor, sample_df):
    """Test target transformation."""
    preprocessor.fit(sample_df)

    target_encoded = preprocessor.transform_target(sample_df["subscribe"])
    assert len(target_encoded) == 3
    assert set(target_encoded) == {0, 1}


def test_preprocessor_inverse_transform_target(preprocessor, sample_df):
    """Test inverse target transformation."""
    preprocessor.fit(sample_df)

    target_encoded = preprocessor.transform_target(sample_df["subscribe"])
    target_original = preprocessor.inverse_transform_target(target_encoded)

    expected = sample_df["subscribe"].values
    np.testing.assert_array_equal(target_original, expected)


def test_preprocessor_get_feature_names(preprocessor, sample_df):
    """Test getting feature names after transformation."""
    preprocessor.fit(sample_df)
    feature_names = preprocessor.get_feature_names()

    assert len(feature_names) > 10
    assert any("job_" in name for name in feature_names)
    assert "age" in feature_names


def test_prepare_prediction_input():
    """Test preparing prediction input."""
    data = {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 400,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": 1.1,
        "cons_price_index": 93.994,
        "cons_conf_index": -36.4,
        "lending_rate3m": 4.85,
        "nr_employed": 5191.0,
    }

    cat_values = {"job": ["admin.", "services"], "default": ["no", "yes"]}
    df = prepare_prediction_input(data, cat_values)

    assert df.shape == (1, 20)
    assert df["age"].iloc[0] == 35
    assert df["job"].iloc[0] == "admin."


def test_prepare_prediction_input_missing_keys():
    """Test error when required keys are missing."""
    data = {"age": 35, "job": "admin."}
    with pytest.raises(ValueError, match="Missing required keys"):
        prepare_prediction_input(data, {})


def test_prepare_prediction_input_invalid_categorical():
    """Test error with invalid categorical value."""
    data = {
        "age": 35,
        "job": "invalid_job",  # Invalid
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 400,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": 1.1,
        "cons_price_index": 93.994,
        "cons_conf_index": -36.4,
        "lending_rate3m": 4.85,
        "nr_employed": 5191.0,
    }

    cat_values = {"job": ["admin.", "services"], "default": ["no", "yes"]}
    with pytest.raises(ValueError, match="Invalid value for job"):
        prepare_prediction_input(data, cat_values)

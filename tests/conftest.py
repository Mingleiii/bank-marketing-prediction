"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "age": 45,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 300,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": 1.1,
        "cons_price_index": 93.994,
        "cons_conf_index": -36.4,
        "lending_rate3m": 4.855,
        "nr_employed": 5191.0,
    }

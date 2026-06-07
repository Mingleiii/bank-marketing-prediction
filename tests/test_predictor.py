"""Tests for model predictor module."""

import pytest

from src.models.predictor import ModelPredictor


@pytest.fixture
def sample_data():
    """Sample data for prediction."""
    return {
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


def test_predictor_initialization_without_model():
    """Test predictor initialization raises error without trained model."""
    with pytest.raises(FileNotFoundError):
        ModelPredictor()


def test_predictor_initialization_with_mock_model(tmp_path, sample_data):
    """Test predictor initialization with a mock model."""
    from sklearn.ensemble import RandomForestClassifier

    from src.utils.data_loader import DataLoader

    # Create and save mock preprocessor
    from src.utils.preprocessing import DataPreprocessor

    loader = DataLoader()
    train_df = loader.load_train().head(100)
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_df)

    # Create and save mock model
    mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
    features, target = preprocessor.fit_transform(train_df.head(50))
    mock_model.fit(features, target)

    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"

    import joblib

    joblib.dump(mock_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    predictor = ModelPredictor(model_path, preprocessor_path)

    assert predictor.model is not None
    assert predictor.preprocessor is not None


def test_predictor_predict(tmp_path, sample_data):
    """Test making a prediction."""
    from sklearn.ensemble import RandomForestClassifier

    from src.utils.data_loader import DataLoader

    # Create and save mock preprocessor
    from src.utils.preprocessing import DataPreprocessor

    loader = DataLoader()
    train_df = loader.load_train().head(100)
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_df)

    # Create and save mock model
    mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
    features, target = preprocessor.fit_transform(train_df.head(50))
    mock_model.fit(features, target)

    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"

    import joblib

    joblib.dump(mock_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    predictor = ModelPredictor(model_path, preprocessor_path)
    result = predictor.predict(sample_data)

    assert "prediction" in result
    assert result["prediction"] in ["yes", "no"]
    assert "subscribe_probability" in result
    assert 0 <= result["subscribe_probability"] <= 1
    assert "not_subscribe_probability" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1


def test_predictor_predict_batch(tmp_path):
    """Test making batch predictions."""
    from sklearn.ensemble import RandomForestClassifier

    from src.utils.data_loader import DataLoader
    from src.utils.preprocessing import DataPreprocessor

    loader = DataLoader()
    train_df = loader.load_train().head(100)
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_df)

    mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
    features, target = preprocessor.fit_transform(train_df.head(50))
    mock_model.fit(features, target)

    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"

    import joblib

    joblib.dump(mock_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    predictor = ModelPredictor(model_path, preprocessor_path)

    data_list = [
        {
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
        },
        {
            "age": 45,
            "job": "services",
            "marital": "single",
            "education": "high.school",
            "default": "no",
            "housing": "no",
            "loan": "yes",
            "contact": "telephone",
            "month": "aug",
            "day_of_week": "tue",
            "duration": 200,
            "campaign": 2,
            "pdays": 10,
            "previous": 1,
            "poutcome": "success",
            "emp_var_rate": -0.1,
            "cons_price_index": 94.0,
            "cons_conf_index": -40.0,
            "lending_rate3m": 4.9,
            "nr_employed": 5200.0,
        },
    ]

    results = predictor.predict_batch(data_list)

    assert len(results) == 2
    for result in results:
        assert "prediction" in result
        assert result["prediction"] in ["yes", "no"]


def test_predictor_get_categorical_options(tmp_path):
    """Test getting categorical options."""
    from sklearn.ensemble import RandomForestClassifier

    from src.utils.data_loader import DataLoader
    from src.utils.preprocessing import DataPreprocessor

    loader = DataLoader()
    train_df = loader.load_train().head(100)
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_df)

    mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
    features, target = preprocessor.fit_transform(train_df.head(50))
    mock_model.fit(features, target)

    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"

    import joblib

    joblib.dump(mock_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    predictor = ModelPredictor(model_path, preprocessor_path)
    options = predictor.get_categorical_options()

    assert "job" in options
    assert "marital" in options
    assert "education" in options
    assert isinstance(options["job"], list)


def test_predictor_get_model_info(tmp_path):
    """Test getting model information."""
    from sklearn.ensemble import RandomForestClassifier

    from src.utils.data_loader import DataLoader
    from src.utils.preprocessing import DataPreprocessor

    loader = DataLoader()
    train_df = loader.load_train().head(100)
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_df)

    mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
    features, target = preprocessor.fit_transform(train_df.head(50))
    mock_model.fit(features, target)

    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"

    import joblib

    joblib.dump(mock_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    predictor = ModelPredictor(model_path, preprocessor_path)
    info = predictor.get_model_info()

    assert "model_type" in info
    assert "model_path" in info
    assert "preprocessor_path" in info
    assert "is_fitted" in info
    assert "RandomForestClassifier" in info["model_type"]

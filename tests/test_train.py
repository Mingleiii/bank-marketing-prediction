"""Tests for model training module."""


def test_model_trainer_initialization():
    """Test model trainer initialization."""
    from src.models.train import ModelTrainer

    trainer = ModelTrainer()
    assert len(trainer.models) == 4
    assert "logistic_regression" in trainer.models
    assert "random_forest" in trainer.models
    assert "gradient_boosting" in trainer.models


def test_model_trainer_load_and_prepare():
    """Test data loading and preparation."""
    from src.models.train import ModelTrainer

    trainer = ModelTrainer()
    train_df, val_df = trainer.load_and_prepare_data()

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert "subscribe" in train_df.columns
    assert "subscribe" in val_df.columns
    assert "id" not in train_df.columns
    assert "id" not in val_df.columns


def test_model_trainer_train_all_models():
    """Test training all models."""
    from src.models.train import ModelTrainer

    trainer = ModelTrainer()
    train_df, val_df = trainer.load_and_prepare_data()

    results = trainer.train_all_models(train_df, val_df)

    assert len(results) == 4
    for _model_name, result in results.items():
        assert "metrics" in result
        assert "model" in result
        assert "auc_roc" in result["metrics"]
        assert 0 <= result["metrics"]["auc_roc"] <= 1


def test_model_trainer_select_best_model():
    """Test selecting the best model."""
    from src.models.train import ModelTrainer

    trainer = ModelTrainer()
    train_df, val_df = trainer.load_and_prepare_data()

    results = trainer.train_all_models(train_df, val_df)
    best_name, best_model, best_preprocessor = trainer.select_best_model(results)

    assert best_name in results
    assert best_model is not None
    assert best_preprocessor is not None
    assert best_preprocessor.is_fitted


def test_model_trainer_save_model(tmp_path):
    """Test saving model and preprocessor."""
    from src.models.train import ModelTrainer

    trainer = ModelTrainer(model_dir=tmp_path)
    train_df, val_df = trainer.load_and_prepare_data()

    results = trainer.train_all_models(train_df, val_df)
    best_name, best_model, best_preprocessor = trainer.select_best_model(results)

    model_path = trainer.save_model(best_model, best_preprocessor, "test_model")

    assert model_path.exists()
    assert (tmp_path / "test_model_preprocessor.pkl").exists()


def test_model_trainer_save_metrics(tmp_path):
    """Test saving metrics."""
    from src.models.train import ModelTrainer

    trainer = ModelTrainer(model_dir=tmp_path)

    test_metrics = {"auc_roc": 0.85, "f1": 0.78}
    metrics_path = trainer.save_metrics(test_metrics, "test_model")

    assert metrics_path.exists()

    import json

    with open(metrics_path) as f:
        loaded_metrics = json.load(f)

    assert loaded_metrics == test_metrics


def test_model_trainer_metrics_above_threshold():
    """Test that at least one model achieves acceptable performance."""
    from src.models.train import ModelTrainer

    trainer = ModelTrainer()
    train_df, val_df = trainer.load_and_prepare_data()

    results = trainer.train_all_models(train_df, val_df)

    # Check that at least one model has AUC-ROC > 0.75
    best_auc = max(r["metrics"]["auc_roc"] for r in results.values())
    assert best_auc > 0.75, f"Best AUC-ROC ({best_auc}) is below threshold (0.75)"

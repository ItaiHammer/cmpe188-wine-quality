import json
from unittest.mock import Mock

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import evaluate
from src import save_model as model_storage


@pytest.fixture
def saved_experiment(tmp_path, monkeypatch):
    """Save a small test experiment."""

    # creating a small dataset
    data = pd.DataFrame({
        "feature_a": [-4, -3, -4, -1, 1, 2, 3, 4],
        "feature_b": [100, 250, 100, 400, 350, 500, 450, 600],
        "good_quality": [0, 0, 0, 0, 1, 1, 1, 1],
    })

    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)
    monkeypatch.setattr(evaluate, "DATA_PATH", str(data_path))
    monkeypatch.setattr(model_storage, "EXPERIMENTS_DIR", str(tmp_path / "runs"))

    # defining feature order and splits
    features = ["feature_b", "feature_a"]
    splits = {"train": [0, 4, 5, 7], "validation": [6, 1], "test": [3]}
    X_train = data.loc[splits["train"], features]
    y_train = data.loc[splits["train"], "good_quality"]

    # fit a scaler and model
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)

    return model_storage.save_model(model, scaler, features, splits, "example")


# tests
def test_metrics_match_known_predictions():
    """Check scores against predetermined ones"""

    model = Mock()
    model.predict.return_value = [0, 0, 0, 1, 0, 0, 1, 1]
    y_val = [0, 0, 0, 0, 1, 1, 1, 1]

    results = evaluate.evaluate_model(model, [[0]] * 8, y_val)

    assert results["split"] == "validation"
    assert results["accuracy"] == pytest.approx(5 / 8)
    assert results["precision"] == pytest.approx(2 / 3)
    assert results["recall"] == pytest.approx(1 / 2)
    assert results["f1"] == pytest.approx(4 / 7)
    assert results["labels"] == [0, 1]
    assert results["confusion_matrix"] == [[3, 1], [2, 2]]


def test_scores_record_the_test_split():
    """Check results are labeled as test when split=test."""

    model = Mock()
    model.predict.return_value = [0, 1]

    results = evaluate.evaluate_model(model, [[0], [1]], [0, 1], split="test")

    assert results["split"] == "test"
    assert results["rows"] == 2
    assert results["accuracy"] == 1


def test_no_positive_predictions():
    """Check precision, recall, and F1 are zero when there are 0 positive predictions."""

    model = Mock()
    model.predict.return_value = [0, 0]

    results = evaluate.evaluate_model(model, [[0], [1]], [0, 1])

    assert results["precision"] == 0
    assert results["recall"] == 0
    assert results["f1"] == 0
    assert results["confusion_matrix"] == [[1, 0], [1, 0]]


def test_validation_rows_and_feature_order(saved_experiment):
    """Check validation rows, feature order, values, and labels match the setup."""

    model, scaler, recipe, splits = evaluate.load_experiment(saved_experiment)

    X_val, y_val = evaluate.load_evaluation_data(recipe, splits)

    assert X_val.index.tolist() == [6, 1]
    assert X_val.columns.tolist() == ["feature_b", "feature_a"]
    assert X_val.to_numpy().tolist() == [[450, 3], [250, -3]]
    assert y_val.tolist() == [1, 0]


def test_saved_evaluation_reuses_fitted_objects(saved_experiment, monkeypatch):
    """Check that evaluation saves the scores without refitting or duplicating README results."""

    # getting scores from the saved model and scaler
    model, scaler, recipe, splits = evaluate.load_experiment(saved_experiment)
    X_val, y_val = evaluate.load_evaluation_data(recipe, splits)
    expected = evaluate.evaluate_model(model, scaler.transform(X_val), y_val)

    # failing if evaluation tries to fit again
    def reject_fit(*args, **kwargs):
        pytest.fail("Evaluation must reuse the fitted model and scaler.")

    monkeypatch.setattr(LogisticRegression, "fit", reject_fit)
    monkeypatch.setattr(StandardScaler, "fit", reject_fit)

    results = evaluate.evaluate_experiment(saved_experiment)

    # checking scores and json
    results_text = (saved_experiment / "validation_metrics.json").read_text(encoding="utf-8")
    assert results == expected
    assert json.loads(results_text) == expected
    assert results["rows"] == 2
    assert results["split"] == "validation"

    # running again should update the README without duplicating the results section
    evaluate.evaluate_experiment(saved_experiment)
    readme = (saved_experiment / "README.md").read_text(encoding="utf-8")
    assert readme.count("## 📈 Validation Results") == 1


@pytest.mark.parametrize("rows", [[], [6, 6], [0], [3], [99]])
def test_invalid_validation_indexes(saved_experiment, rows):
    """Reject invalid validation rows."""

    model, scaler, recipe, splits = evaluate.load_experiment(saved_experiment)
    splits["validation"] = rows

    with pytest.raises((ValueError, KeyError)):
        evaluate.load_evaluation_data(recipe, splits)


def test_test_rows_and_feature_order(saved_experiment):
    """Check split=test loads the saved test rows with the correct feature order, values, and labels."""

    model, scaler, recipe, splits = evaluate.load_experiment(saved_experiment)

    X_eval, y_eval = evaluate.load_evaluation_data(recipe, splits, split="test")

    assert X_eval.index.tolist() == [3]
    assert X_eval.columns.tolist() == ["feature_b", "feature_a"]
    assert X_eval.to_numpy().tolist() == [[400, -1]]
    assert y_eval.tolist() == [0]


def test_test_rows_cannot_overlap_validation(saved_experiment):
    """Checking that a row cannot belong to both test and validation."""

    model, scaler, recipe, splits = evaluate.load_experiment(saved_experiment)
    splits["test"] = [6]

    with pytest.raises(ValueError):
        evaluate.load_evaluation_data(recipe, splits, split="test")


def test_training_split_is_not_an_evaluation_choice():
    """Allow only validation or test as values for split."""

    with pytest.raises(ValueError):
        evaluate.load_evaluation_data({}, {}, split="train")

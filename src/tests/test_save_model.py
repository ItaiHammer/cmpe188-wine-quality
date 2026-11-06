import json

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

# src
from src import save_model as model_storage
from src.save_model import save_model, validate_experiment_name
from src.train import build_model


@pytest.fixture
def training_data(tmp_path, monkeypatch):
    """Provide a small dataset and redirect saves to a temporary folder."""
    X = np.array([
        [-3, 1],
        [-2, 2],
        [-1, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [-1.5, 3.5],
        [1.5, 4.5],
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])

    scaler = StandardScaler().fit(X[:6])
    features = ["feature_a", "feature_b"]
    splits = {"train": list(range(6)), "validation": [6], "test": [7]}

    monkeypatch.setattr(model_storage, "EXPERIMENTS_DIR", f"{tmp_path}/local_experiments")
    return X, y, scaler, features, splits


# tests
@pytest.mark.parametrize("model_name", ["logistic", "forest", "svm"])
def test_saved_model_and_recipe(model_name, training_data):
    """Saving and reloading preserves predictions, settings, and split rows."""
    X, y, scaler, features, splits = training_data

    # training
    model = build_model(model_name)
    model.fit(scaler.transform(X[:6]), y[:6])
    expected_predictions = model.predict(scaler.transform(X[6:]))

    # saving and loading
    run_dir = save_model(model, scaler, features, splits, model_name)
    saved_model = joblib.load(run_dir / "model.joblib")
    saved_scaler = joblib.load(run_dir / "scaler.joblib")
    recipe_text = (run_dir / "recipe.json").read_text(encoding="utf-8")
    saved_recipe = json.loads(recipe_text)
    split_text = (run_dir / "split_indices.json").read_text(encoding="utf-8")
    saved_splits = json.loads(split_text)

    # checking saved data
    scaled_inputs = saved_scaler.transform(X[6:])
    actual_predictions = saved_model.predict(scaled_inputs)
    np.testing.assert_array_equal(actual_predictions, expected_predictions)

    assert saved_recipe["feature_names"] == features
    assert saved_recipe["params"] == model.get_params()
    assert saved_recipe["scaler"]["params"] == scaler.get_params()
    assert saved_splits == splits
    assert (run_dir / "README.md").is_file()


def test_existing_run_is_not_overwritten(training_data):
    """Reusing a run name raises an error and leaves its files unchanged."""
    X, y, scaler, features, splits = training_data
    model = build_model("logistic")
    model.fit(scaler.transform(X[:6]), y[:6])
    run_dir = save_model(model, scaler, features, splits, "first-run")

    original_files = {}
    for file_path in run_dir.iterdir():
        original_files[file_path.name] = file_path.read_bytes()

    with pytest.raises(FileExistsError):
        validate_experiment_name("first-run")

    with pytest.raises(FileExistsError):
        save_model(model, scaler, features, splits, "first-run")

    for file_name, original_contents in original_files.items():
        assert (run_dir / file_name).read_bytes() == original_contents


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../outside", "nested/run", r"..\outside", "/outside"],
)
def test_invalid_names_cannot_select_other_directories(name):
    """Reject empty names and names that point outside a single run folder."""
    with pytest.raises(ValueError):
        validate_experiment_name(name)

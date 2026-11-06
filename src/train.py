import argparse
import json
from contextlib import redirect_stdout
from io import StringIO

from rich.console import Console
from rich.text import Text

# model classes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# src
from src.preprocessing import prepare_data
from src.save_model import save_model, validate_experiment_name
from src.evaluate import evaluate_experiment


console = Console(highlight=False)


def build_model(name, **params):
    """Create a model instance from parameters."""
    model_types = {
        "logistic": LogisticRegression,
        # default params:
        # {
        #     "penalty": "deprecated",
        #     "C": 1.0,
        #     "l1_ratio": 0.0,
        #     "dual": false,
        #     "tol": 0.0001,
        #     "fit_intercept": true,
        #     "intercept_scaling": 1,
        #     "class_weight": null,
        #     "random_state": null,
        #     "solver": "lbfgs",
        #     "max_iter": 100,
        #     "verbose": 0,
        #     "warm_start": false,
        #     "n_jobs": null
        # }

        "forest": RandomForestClassifier,
        # default params:
        # {
        #     "n_estimators": 100,
        #     "criterion": "gini",
        #     "max_depth": null,
        #     "min_samples_split": 2,
        #     "min_samples_leaf": 1,
        #     "min_weight_fraction_leaf": 0.0,
        #     "max_features": "sqrt",
        #     "max_leaf_nodes": null,
        #     "min_impurity_decrease": 0.0,
        #     "bootstrap": true,
        #     "oob_score": false,
        #     "n_jobs": null,
        #     "random_state": null,
        #     "verbose": 0,
        #     "warm_start": false,
        #     "class_weight": null,
        #     "ccp_alpha": 0.0,
        #     "max_samples": null,
        #     "monotonic_cst": null
        # }

        "svm": SVC,
        # default params:
        # {
        #     "C": 1.0,
        #     "kernel": "rbf",
        #     "degree": 3,
        #     "gamma": "scale",
        #     "coef0": 0.0,
        #     "shrinking": true,
        #     "probability": "deprecated",
        #     "tol": 0.001,
        #     "cache_size": 200,
        #     "class_weight": null,
        #     "verbose": false,
        #     "max_iter": -1,
        #     "decision_function_shape": "ovr",
        #     "break_ties": false,
        #     "random_state": null
        # }
    }

    if name not in model_types:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {', '.join(model_types)}"
        )

    return model_types[name](**params)


def train_model(model, X_train, y_train):
    """Train the model on data using X_train and y_train and return the fitted model."""
    model.fit(X_train, y_train)
    return model


def main():
    """Read cli command with settings and train the model."""

    # handling args
    parser = argparse.ArgumentParser(description="Train a wine quality model.")
    parser.add_argument("--model", required=True, choices=["logistic", "forest", "svm"])
    parser.add_argument("--name", required=True, help="Unique experiment folder name.")
    parser.add_argument("--params", default="{}", help="Model settings as a JSON object.")
    args = parser.parse_args()

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as error:
        parser.error(f"--params must be valid JSON: {error.msg}")

    if not isinstance(params, dict):
        parser.error("--params must be a JSON object, such as {\"C\": 1.0}")


    # checking experiment name overlap
    try:
        validate_experiment_name(args.name)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))

    # building model
    try:
        model = build_model(args.model, **params)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    console.rule(Text("Training", style="white"), style="white")
    console.print(f"Experiment Name: {args.name}  |  Model: {args.model}", style="dim", markup=False)
    console.print("[1/4] Preparing data...", style="white", markup=False)

    # ignoring stdout from preprocessing
    with redirect_stdout(StringIO()):
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler, feature_names,
        ) = prepare_data() # from src.preprocessing.py
    console.print(
        f"      Train: {len(y_train):,}  |  Validation: {len(y_val):,}  |  Test: {len(y_test):,}",
        style="dim",
    )

    console.print(f"[2/4] Training {args.model}...", style="white", markup=False)
    with console.status("Fitting model...", spinner="dots", spinner_style="white"):
        trained_model = train_model(model, X_train, y_train)
    console.print("      Training complete.", style="green")

    split_indices = {
        "train": y_train.index.tolist(),
        "validation": y_val.index.tolist(),
        "test": y_test.index.tolist(),
    }

    try:
        console.print("[3/4] Saving experiment...", style="white", markup=False)

        # from src/save_model.py
        experiment_dir = save_model(
            trained_model,
            scaler,
            feature_names,
            split_indices,
            args.name,
        )
    except OSError as error:
        parser.error(str(error))

    console.print("[4/4] Evaluating on validation set...", style="white", markup=False)
    try:
        evaluate_experiment(experiment_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"Model saved to {experiment_dir}, but validation evaluation failed: {error}")

    console.rule("Complete", style="green")
    console.print("Experiment folder:", Text(str(experiment_dir), style="blue"), style="white")
    return trained_model


if __name__ == "__main__":
    main()

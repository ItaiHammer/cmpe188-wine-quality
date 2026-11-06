import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# src
from src.save_model import save_readme


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = f"{PROJECT_ROOT}/data/10_wine_quality_train.csv"
console = Console(highlight=False)


def load_experiment(experiment_dir):
    """Loads the model, scaler, recipe, and split indexes."""

    experiment_dir = Path(experiment_dir)

    # load all the files
    model = joblib.load(f"{experiment_dir}/model.joblib")
    scaler = joblib.load(f"{experiment_dir}/scaler.joblib")
    recipe = json.loads(Path(f"{experiment_dir}/recipe.json").read_text(encoding="utf-8"))
    splits = json.loads(Path(f"{experiment_dir}/split_indices.json").read_text(encoding="utf-8"))

    return model, scaler, recipe, splits


def load_evaluation_data(recipe, splits, split="validation"):
    """Loads validation or test rows in the saved feature order."""

    if split not in ("validation", "test"):
        raise ValueError("Choose either 'validation' or 'test' for evaluation.")

    rows = splits[split]
    other_split = "test" if split == "validation" else "validation"

    # checking split row indexes
    if not rows:
        raise ValueError(f"The saved {split} split is empty.")

    if len(rows) != len(set(rows)):
        raise ValueError(f"The saved {split} split contains repeated row indexes.")

    if set(rows).intersection(splits["train"] + splits[other_split]):
        raise ValueError(f"{split.title()} rows overlap the training or {other_split} split.")

    # loading
    data = pd.read_csv(DATA_PATH)
    X_eval = data.loc[rows, recipe["feature_names"]]
    y_eval = data.loc[rows, "good_quality"]

    return X_eval, y_eval


def evaluate_model(model, X_eval, y_eval, split="validation"):
    """Calculate scores for the selected split; good_quality=1 is positive."""

    predictions = model.predict(X_eval)
    return {
        "split": split,
        "rows": len(y_eval),
        "positive_label": 1,
        "accuracy": float(accuracy_score(y_eval, predictions)),
        "precision": float(precision_score(y_eval, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_eval, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_eval, predictions, pos_label=1, zero_division=0)),
        "labels": [0, 1],
        "confusion_matrix": confusion_matrix(y_eval, predictions, labels=[0, 1]).tolist(),
    }


def print_results(results):
    """Print out the scores and confusion matrix for the selected split."""
    split_title = results["split"].title()

    # printing scores in a table
    console.rule(split_title, style="white")
    console.print(f"Rows: {results['rows']:,}  |  Positive class: good_quality = 1", style="dim")

    scores = Table(
        title=f"{split_title} metrics", title_justify="left", title_style="white", border_style="dim"
    )
    scores.add_column("Metric")
    scores.add_column("Score", justify="right")

    for name in ("accuracy", "precision", "recall", "f1"):
        scores.add_row(name.title(), f"{results[name]:.4f}")

    console.print(scores)

    # print out the confusion matrix
    matrix = Table(
        title=f"{split_title} confusion matrix", title_justify="left", title_style="white", border_style="dim"
    )
    matrix.add_column("Actual / predicted")
    matrix.add_column("0", justify="right")
    matrix.add_column("1", justify="right")

    for label, row in zip(results["labels"], results["confusion_matrix"]):
        matrix.add_row(str(label), str(row[0]), str(row[1]))

    console.print(matrix)


def evaluate_experiment(experiment_dir, split="validation"):
    """Evaluate a saved experiment on the selected split and update its README."""

    # load data
    experiment_dir = Path(experiment_dir).resolve()
    model, scaler, recipe, splits = load_experiment(experiment_dir)
    X_eval, y_eval = load_evaluation_data(recipe, splits, split=split)

    # scale inputs and calculate scores
    scaled_inputs = scaler.transform(X_eval)
    results = evaluate_model(model, scaled_inputs, y_eval, split=split)

    # save
    results_path = Path(f"{experiment_dir}/{split}_metrics.json")
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_readme(recipe, splits, experiment_dir)

    # print
    print_results(results)
    console.print(f"  + {split}_metrics.json", style="green")
    console.print("  Updated README.md", style="green")

    return results


def main():
    """Evaluate a saved experiment from the command line."""

    # args
    parser = argparse.ArgumentParser(description="Evaluate a saved wine model on validation or test.")
    parser.add_argument("--experiment", required=True, help="Saved experiment folder.")
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation",
        help="Split to evaluate (default: validation).",
    )
    args = parser.parse_args()

    # arg validation
    try:
        return evaluate_experiment(args.experiment, split=args.split)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

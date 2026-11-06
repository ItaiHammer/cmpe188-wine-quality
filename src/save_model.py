import json
from pathlib import Path
import joblib
from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = f"{PROJECT_ROOT}/local_experiments"
console = Console(highlight=False)

def validate_experiment_name(name):
    """Check that the experiment name is valid."""

    if not name:
        raise ValueError("Run name cannot be empty.")

    if not name[0].isalnum():
        raise ValueError("Run names must start with a letter or number.")

    # illegal character check
    only_letters_and_numbers = name.replace("-", "").replace("_", "").replace(".", "")
    if not only_letters_and_numbers.isalnum():
        raise ValueError("Use only letters, numbers, '.', '_' or '-' in run names.")

    experiment_dir = Path(f"{EXPERIMENTS_DIR}/{name}")
    if experiment_dir.exists():
        raise FileExistsError(f"Experiment '{name}' already exists. Choose a new name.")


def save_readme(recipe, split_indices, experiment_dir, validation_metrics=None, test_metrics=None):
    """Write an experiment summary with any available validation and test results."""
    parameter_rows = "\n".join(
        f"| `{name}` | `{json.dumps(value)}` |"
        for name, value in recipe["params"].items()
    )
    scaler_rows = "\n".join(
        f"| `{name}` | `{json.dumps(value)}` |"
        for name, value in recipe["scaler"]["params"].items()
    )
    feature_rows = "\n".join(
        f"| {index} | `{feature}` |"
        for index, feature in enumerate(recipe["feature_names"], start=1)
    )
    evaluation_files = ""
    evaluation_sections = ""

    # keeping results from either split when the README is updated
    for split, metrics in (("validation", validation_metrics), ("test", test_metrics)):
        metrics_path = Path(f"{experiment_dir}/{split}_metrics.json")
        if metrics is None and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        if metrics is None:
            continue

        score_rows = "\n".join(
            f"| {name.title()} | {metrics[name]:.4f} |"
            for name in ("accuracy", "precision", "recall", "f1")
        )
        matrix = metrics["confusion_matrix"]
        evaluation_files += f"├── {split}_metrics.json    # {split} scores and confusion matrix\n"
        evaluation_sections += f'''## 📈 {split.title()} Results

Evaluated **{metrics["rows"]:,} {split} rows**. Positive class: `good_quality = 1`.

| Metric | Score |
| :----- | ----: |
{score_rows}

| Actual / predicted | 0 | 1 |
| :----------------- | -: | -: |
| 0 | {matrix[0][0]} | {matrix[0][1]} |
| 1 | {matrix[1][0]} | {matrix[1][1]} |

Undefined precision, recall, or F1 is reported as 0.
Full results are saved in [{split}_metrics.json]({split}_metrics.json).

'''
    readme = f'''# 🍷 {experiment_dir.name}

**Wine Quality Classification**

`{recipe["model_type"]}` trained to predict `good_quality` from wine chemistry measurements.

## 🧠 Model

| Algorithm | Preprocessing | Input features |
| :-------- | :------------ | -------------: |
| `{recipe["model_type"]}` | `{recipe["scaler"]["type"]}` | {len(recipe["feature_names"])} |

## 📊 Data Splits

| Training | Validation | Test |
| -------: | ---------: | ---: |
| **{len(split_indices["train"]):,} rows** | **{len(split_indices["validation"]):,} rows** | **{len(split_indices["test"]):,} rows** |

The model and scaler were fitted using the training split. Exact row indexes are stored in `split_indices.json`.

## ⚙️ Configuration

| Parameter | Value |
| :-------- | :---- |
{parameter_rows}

Includes the supplied parameters and the remaining model defaults.

## 🧪 Preprocessing

Reuse `scaler.joblib` when preparing inputs for this model.

| Scaler setting | Value |
| :------------- | :---- |
{scaler_rows}

**Input feature order**

| # | Feature |
| -: | :------ |
{feature_rows}

## 📂 Run Files

```text
{experiment_dir.name}/
├── model.joblib          # fitted model
├── scaler.joblib         # fitted scaler
├── recipe.json           # feature names and model/scaler settings
├── split_indices.json    # train, validation, and test row indexes
{evaluation_files}└── README.md             # this summary
```

{evaluation_sections}
'''
    readme_path = Path(f"{experiment_dir}/README.md")
    readme_path.write_text(readme, encoding="utf-8")


def save_model(model, scaler, feature_names, split_indices, name):
    """Save a model and its recipe to evaluate it later."""

    # compiling all the info we wanna store in the model folder
    experiment_dir = Path(f"{EXPERIMENTS_DIR}/{name}")
    recipe = {
        "model_type": type(model).__name__,
        "scaler": {
            "type": type(scaler).__name__,
            "params": scaler.get_params(),
        },
        "feature_names": list(feature_names),
        "params": model.get_params(),
    }

    # creating the directory and file paths
    experiment_dir.mkdir(parents=True, exist_ok=False)
    model_path = Path(f"{experiment_dir}/model.joblib")
    scaler_path = Path(f"{experiment_dir}/scaler.joblib")
    recipe_path = Path(f"{experiment_dir}/recipe.json")
    split_path = Path(f"{experiment_dir}/split_indices.json")

    # writing to them
    joblib.dump(model, model_path, compress=3)
    console.print("  + model.joblib", style="green")

    joblib.dump(scaler, scaler_path, compress=3)
    console.print("  + scaler.joblib", style="green")

    recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    console.print("  + recipe.json", style="green")

    split_path.write_text(json.dumps(split_indices, indent=2), encoding="utf-8")
    console.print("  + split_indices.json", style="green")

    save_readme(recipe, split_indices, experiment_dir)
    console.print("  + README.md", style="green")

    return experiment_dir

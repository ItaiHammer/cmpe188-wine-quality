<h1 align="center">🍷 Wine Quality Classification</h1>

<p align="center">
  <strong>CMPE 188 Class Project</strong><br>
  CMPE 188 · Team 11 · San José State University · Fall 2026
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&amp;logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&amp;logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?logo=googlecolab&amp;logoColor=white" alt="Google Colab">
</p>

---

## 🎯 About

The project's goal is to create a model to predict whether a wine is good quality from its chemical properties. The project will cover preprocessing, model comparison, fairness, interpretability, and a final presentation.

| Quality score | Binary target      |
| :------------ | :----------------- |
| ≥ 6           | `good_quality = 1` |
| < 6           | `good_quality = 0` |

> **Current stage:** Model training, saving, and evaluation are implemented.

## 📊 Dataset

[`data/10_wine_quality_train.csv`](data/10_wine_quality_train.csv)

| Samples | Input features | Good quality (1) | Lower quality (0) |
| :-----: | :------------: | :--------------: | :---------------: |
|  5,522  |   11 numeric   |      3,496       |       2,026       |

The binary label is already supplied; the original `quality` score is absent. Keep the CSV unchanged and exclude `good_quality` from model inputs.

## 📂 Project Structure

```text
├── data/                          # given training data and documentation
│   ├── 10_wine_quality_train.csv
│   └── README.md                  # ideas and observations about the data
├── notebooks/                     # Google Colab notebooks (empty for now)
├── src/                           # reusable Python code
│   ├── evaluate.py                # model evaluation
│   ├── preprocessing.py           # data preparation
│   ├── save_model.py              # save models, recipes, and split indexes
│   ├── tests/                     # tests for src files
│   └── train.py                   # model creation and training
├── local_experiments/             # local model experiments (ignored by git)
├── experiments/                   # experiments we want to keep
├── .gitignore
├── pytest.ini                     # pytest settings
├── requirements.txt               # dependencies
├── requirements-dev.txt           # test dependencies
└── README.md
```

## 🚀 Getting Started

Use `Python 3.11+`. Create the venv once from the repo root, then activate it whenever you open a new terminal:

Windows and WSL need separate virtual environments.

**Linux/macOS/WSL**

```bash
python3 -m venv .venv        # create the venv
source .venv/bin/activate   # enter the venv
```

**Windows (PowerShell)**

```powershell
python -m venv venv             # create the venv
.\venv\Scripts\Activate.ps1     # enter the venv
```

After activating, choose one:

| Setup       | Includes                    | Install command                                 |
| :---------- | :-------------------------- | :---------------------------------------------- |
| Training    | Training libraries          | `python -m pip install -r requirements.txt`     |
| Development | Training libraries + pytest | `python -m pip install -r requirements-dev.txt` |

> The development install includes the training dependencies. You only need one install.

To leave the environment, run `deactivate`.

## 🧠 Training

Run [`src/train.py`](src/train.py) as a module from the root:

| Flag       | Required | Values                                                                          |
| :--------- | :------: | :------------------------------------------------------------------------------ |
| `--model`  |   Yes    | `logistic`, `forest`, or `svm`                                                  |
| `--name`   |   Yes    | A unique name for your model; ex: `logistic-v1`                                 |
| `--params` |    No    | Model settings as JSON; parameters you don't provide use scikit-learn defaults. |

Ex: Logistic regression model

**Windows (PowerShell)**

```powershell
python --% -m src.train --model logistic --name logistic-v1 --params "{\"C\": 0.5, \"max_iter\": 1000}"
```

`--%` and the escaped quotes keep PowerShell from changing the JSON argument.

**Linux/macOS/WSL**

```bash
python -m src.train --model logistic --name logistic-v1 --params '{"C": 0.5, "max_iter": 1000}'
```

For more help:

```bash
python -m src.train --help
```

Training shows colored stages, a spinner while fitting, and the files saved for the run.

Each run creates its own folder:

```text
local_experiments/
└── <experiment_name>/             # experiment name
    ├── model.joblib               # model file
    ├── scaler.joblib              # fitted scaler
    ├── recipe.json                # feature names and model/scaler settings
    ├── split_indices.json         # train, validation, and test row indexes
    ├── validation_metrics.json    # validation scores and confusion matrix
    ├── test_metrics.json          # added after evaluation with --split test
    └── README.md                  # folder summary
```

The `local_experiments/` folder is ignored by git. Copy experiment folders into `experiments/` if you want to keep and commit them.

## 📊 Evaluation

Run from the repository root:

| Flag           | Required | Values                            |
| :------------- | :------: | :-------------------------------- |
| `--experiment` |   Yes    | Saved experiment folder.          |
| `--split`      |    No    | `validation` (default) or `test`. |

**Validation**

Ex:

```bash
python -m src.evaluate --experiment local_experiments/logistic-v1
```

**Final test evaluation**

Ex:

```bash
python -m src.evaluate --experiment local_experiments/logistic-v1 --split test
```

Evaluation reports accuracy, precision, recall, F1, and a confusion matrix. Each split has its own json file containing its metrics, and both json files get added to the experiment README.

## 🧪 Tests

**Test files**

| File                                                 | Tests                                                           |
| :--------------------------------------------------- | :-------------------------------------------------------------- |
| [`test_save_model.py`](src/tests/test_save_model.py) | Model saving, overwrite protection, and run name validation.    |
| [`test_evaluate.py`](src/tests/test_evaluate.py)     | Metrics, validation/test row loading, and saved-run evaluation. |

**Run tests**

With development dependencies installed and venv active, run from the repository root:

```bash
python -m pytest
```

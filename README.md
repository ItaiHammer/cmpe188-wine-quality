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

> **Current stage:** Repository setup. No analysis, trained models, or evaluation results are included yet.

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
│   ├── preprocessing.py           # data preparation (placeholder)
│   └── train.py                   # model training (placeholder)
├── experiments/                   # model settings and results (empty for now)
├── .gitignore
└── README.md
```

## 🚀 Getting Started

Nothing here yet... 🦗

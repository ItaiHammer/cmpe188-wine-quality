# 🍷 svm-balanced-candidate-v1

**Wine Quality Classification**

`SVC` trained to predict `good_quality` from wine chemistry measurements.

## 🧠 Model

| Algorithm | Preprocessing | Input features |
| :-------- | :------------ | -------------: |
| `SVC` | `StandardScaler` | 11 |

## 📊 Data Splits

| Training | Validation | Test |
| -------: | ---------: | ---: |
| **2,792 rows** | **931 rows** | **931 rows** |

The model and scaler were fitted using the training split. Exact row indexes are stored in `split_indices.json`.

## ⚙️ Configuration

| Parameter | Value |
| :-------- | :---- |
| `C` | `2.0` |
| `break_ties` | `false` |
| `cache_size` | `200` |
| `class_weight` | `"balanced"` |
| `coef0` | `0.0` |
| `decision_function_shape` | `"ovr"` |
| `degree` | `3` |
| `gamma` | `0.1` |
| `kernel` | `"rbf"` |
| `max_iter` | `-1` |
| `probability` | `"deprecated"` |
| `random_state` | `null` |
| `shrinking` | `true` |
| `tol` | `0.001` |
| `verbose` | `false` |

Includes the supplied parameters and the remaining model defaults.

## 🧪 Preprocessing

Reuse `scaler.joblib` when preparing inputs for this model.

| Scaler setting | Value |
| :------------- | :---- |
| `copy` | `true` |
| `with_mean` | `true` |
| `with_std` | `true` |

**Input feature order**

| # | Feature |
| -: | :------ |
| 1 | `fixed_acidity` |
| 2 | `volatile_acidity` |
| 3 | `citric_acid` |
| 4 | `residual_sugar` |
| 5 | `chlorides` |
| 6 | `free_sulfur_dioxide` |
| 7 | `total_sulfur_dioxide` |
| 8 | `density` |
| 9 | `pH` |
| 10 | `sulphates` |
| 11 | `alcohol` |

## 📂 Run Files

```text
svm-balanced-candidate-v1/
├── model.joblib          # fitted model
├── scaler.joblib         # fitted scaler
├── recipe.json           # feature names and model/scaler settings
├── split_indices.json    # train, validation, and test row indexes
├── validation_metrics.json    # validation scores and confusion matrix
└── README.md             # this summary
```

## 📈 Validation Results

Evaluated **931 validation rows**. Positive class: `good_quality = 1`.

| Metric | Score |
| :----- | ----: |
| Accuracy | 0.7648 |
| Precision | 0.8411 |
| Recall | 0.7705 |
| F1 | 0.8043 |

| Actual / predicted | 0 | 1 |
| :----------------- | -: | -: |
| 0 | 262 | 85 |
| 1 | 134 | 450 |

Undefined precision, recall, or F1 is reported as 0.
Full results are saved in [validation_metrics.json](validation_metrics.json).



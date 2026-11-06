# 🍷 forest-500-candidate-v1

**Wine Quality Classification**

`RandomForestClassifier` trained to predict `good_quality` from wine chemistry measurements.

## 🧠 Model

| Algorithm | Preprocessing | Input features |
| :-------- | :------------ | -------------: |
| `RandomForestClassifier` | `StandardScaler` | 11 |

## 📊 Data Splits

| Training | Validation | Test |
| -------: | ---------: | ---: |
| **2,792 rows** | **931 rows** | **931 rows** |

The model and scaler were fitted using the training split. Exact row indexes are stored in `split_indices.json`.

## ⚙️ Configuration

| Parameter | Value |
| :-------- | :---- |
| `bootstrap` | `true` |
| `ccp_alpha` | `0.0` |
| `class_weight` | `null` |
| `criterion` | `"gini"` |
| `max_depth` | `null` |
| `max_features` | `"sqrt"` |
| `max_leaf_nodes` | `null` |
| `max_samples` | `null` |
| `min_impurity_decrease` | `0.0` |
| `min_samples_leaf` | `1` |
| `min_samples_split` | `2` |
| `min_weight_fraction_leaf` | `0.0` |
| `monotonic_cst` | `null` |
| `n_estimators` | `500` |
| `n_jobs` | `2` |
| `oob_score` | `false` |
| `random_state` | `42` |
| `verbose` | `0` |
| `warm_start` | `false` |

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
forest-500-candidate-v1/
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
| Accuracy | 0.7777 |
| Precision | 0.7941 |
| Recall | 0.8716 |
| F1 | 0.8310 |

| Actual / predicted | 0 | 1 |
| :----------------- | -: | -: |
| 0 | 215 | 132 |
| 1 | 75 | 509 |

Undefined precision, recall, or F1 is reported as 0.
Full results are saved in [validation_metrics.json](validation_metrics.json).



# California Residential Close Price Prediction

An end-to-end machine-learning project that predicts `ClosePrice` for California single-family homes using CRMLS records supplied through IDX Exchange. The workflow covers exploration, leakage-controlled preprocessing, chronological validation, feature engineering, advanced modeling, and segment-level evaluation.

The model is intended for pricing review and prioritization. It is not a formal appraisal and should not replace professional judgment.

## Scope and evaluation design

The analysis includes only:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`
- transactions from May 2025 through June 2026

| Split | Period | Full rows | Primary in-range rows |
|---|---|---:|---:|
| Training | May 2025-April 2026 | 118,774 | 116,424 |
| Validation | May 2026 | 11,984 | 11,735 |
| Test | June 2026 | 12,827 | 12,543 |

The primary population uses the training set's 0.5th and 99.5th percentile limits for `ClosePrice` and price per square foot: $190,000-$8,750,000 and $164-$2,127 per square foot. The same frozen limits are applied to validation and test. Complete June performance is reported separately as a robustness check.

`ListPrice`, `OriginalListPrice`, target-derived predictors, identifiers, agent/office fields, and post-close information are excluded.

## Weekly workflow

| Week | Notebook | Purpose | Detailed results |
|---|---|---|---|
| 2 | [`01_exploration.ipynb`](01_exploration.ipynb) | Explore data quality and distributions | [`docs/01_exploration_README.md`](docs/01_exploration_README.md) |
| 3 | [`02_preprocessing.ipynb`](02_preprocessing.ipynb) | Prepare chronological model datasets | [`docs/02_preprocessing_README.md`](docs/02_preprocessing_README.md) |
| 4 | [`03_baseline_model.ipynb`](03_baseline_model.ipynb) | Establish Dummy and Linear baselines | [`docs/03_baseline_model_README.md`](docs/03_baseline_model_README.md) |
| 5 | [`04_model_comparison.ipynb`](04_model_comparison.ipynb) | Compare baseline model families | [`docs/04_model_comparison_README.md`](docs/04_model_comparison_README.md) |
| 6 | [`03_baseline_model_updated.ipynb`](03_baseline_model_updated.ipynb) | Test engineered features with Linear Regression | [`docs/03_baseline_model_updated_README.md`](docs/03_baseline_model_updated_README.md) |
| 6 | [`04_model_comparison_updated.ipynb`](04_model_comparison_updated.ipynb) | Test engineered features across models | [`docs/04_model_comparison_updated_README.md`](docs/04_model_comparison_updated_README.md) |
| 7 | [`05_advanced_models.ipynb`](05_advanced_models.ipynb) | Tune XGBoost and select the final model | [`docs/05_advanced_models_README.md`](docs/05_advanced_models_README.md) |
| 8 | [`06_evaluation.ipynb`](06_evaluation.ipynb) | Evaluate stability, segments, and review risk | [`docs/06_evaluation_README.md`](docs/06_evaluation_README.md) |

Week 6 intentionally uses two updated notebooks. Together they provide the required old-versus-new feature comparison, including the school-district layer.

## Model progression

### Original feature workflow

| Model | Validation R2 | Validation MdAPE | Primary June R2 | Primary June MdAPE |
|---|---:|---:|---:|---:|
| Median Dummy | -0.1196 | 39.38% | -0.1138 | 39.38% |
| Linear Regression | 0.6110 | 27.16% | 0.6205 | 26.97% |
| Decision Tree | 0.8216 | 10.24% | 0.8220 | 10.20% |
| Random Forest | **0.8872** | **7.82%** | **0.8847** | **8.01%** |

Random Forest substantially outperforms the linear and single-tree baselines.

### Week 6 feature engineering

The expanded set adds bedroom-to-bathroom ratio, property age, living-area-to-lot-area ratio, amenity count, cyclical month features, and spatially assigned school district. District match coverage is 99.99% in training and 100% in validation and test.

Feature engineering improves Linear Regression primary June R2 from 0.8465 to 0.8525 and MdAPE from 15.19% to 14.71%. In the controlled Random Forest experiment, however, the original set performs slightly better: primary June R2 is 0.8811 versus 0.8758. The engineered variables therefore add useful linear structure but do not automatically improve an already nonlinear model.

### Advanced models

Time-ordered cross-validation selects XGBoost depth 7, learning rate 0.10, and 300 estimators.

| Model | Validation R2 | Validation MAE | Validation MdAPE | Primary June R2 | Primary June MAE | Primary June MdAPE |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8757 | **$171,103** | **8.20%** | 0.8762 | **$170,135** | **8.23%** |
| XGBoost | **0.8866** | $175,045 | 9.67% | **0.8906** | $174,249 | 9.68% |

XGBoost has higher R2 and lower RMSE, while Random Forest has lower MAE, MAPE, and MdAPE. Because validation MdAPE is the predefined primary selection metric, Random Forest is selected as the final model.

## Final Random Forest evaluation

| Population | Rows | R2 | MAE | RMSE | MAPE | MdAPE | Within 20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 11,735 | 0.8757 | $171,103 | $334,436 | 12.72% | 8.20% | 81.3% |
| Primary June | 12,543 | **0.8762** | **$170,135** | $335,700 | 12.77% | 8.23% | 81.1% |
| Full June robustness | 12,827 | 0.5999 | $231,306 | $955,058 | 22.72% | 8.48% | 79.4% |

Validation and primary June are highly stable: R2 changes by +0.0004 and MdAPE by only +0.03 percentage points. Complete-June R2 and RMSE deteriorate because a small number of unusual transactions generate very large squared errors, while the typical proportional error remains much more stable.

## Segment findings

- The $500K-$750K band has the lowest MdAPE at 6.05%.
- The $750K-$1M band has the lowest MAPE at 10.10% and 87.9% of predictions within 20%.
- Properties above $2M have MAE of $538,028 and MdAPE of 13.31%.
- Merced, Riverside, and San Bernardino have the lowest county-level MdAPE among counties with at least 100 test rows.
- Santa Cruz has the highest qualifying county MdAPE at 20.76%.
- Underpredictions represent 47.9% and overpredictions 52.1%, indicating limited aggregate directional imbalance.

## Reproduction

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
python -m pip install jupyter geopandas shapely matplotlib seaborn
```

Provide the monthly `CRMLSSoldYYYYMM.csv` files and the California school-district boundary file, update the configuration paths, and run the notebooks in table order. Raw CRMLS data and trained artifacts are not committed because they are externally provided and may be subject to licensing restrictions.

## Limitations

- Full-market performance is weaker for luxury and unusual transactions.
- One validation month and one test month are insufficient for production approval.
- School-district membership is geographic context, not a causal measure of school quality.
- `FireplacesTotal` was entirely missing in this run and was skipped by the imputer.
- Frequency and one-hot encodings can lose information for rare locations.
- Model performance should be monitored as inventory and market conditions change.

## Business recommendation

Use Random Forest as decision support for typical single-family homes, particularly in the $500K-$1M range. Route luxury homes, sparse geographies, unusual configurations, and predictions with elevated estimated risk to manual comparable-sales review.

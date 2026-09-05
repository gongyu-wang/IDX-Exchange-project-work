# Week 7 - Advanced Models

**Notebook:** [`05_advanced_models.ipynb`](../05_advanced_models.ipynb)

## Objective

Tune XGBoost using time-ordered cross-validation and compare it with the Random Forest benchmark. Model-family selection is based on May validation MdAPE and MAE; June is reserved for final assessment.

## Data and features

- Training: May 2025-April 2026, 116,424 primary rows
- Validation: May 2026, 11,735 primary rows
- Test: June 2026, 12,543 primary rows
- Updated input fields: 39
- Complete June robustness population: 12,827 rows

All models exclude `ListPrice`, `OriginalListPrice`, target-derived predictors, identifiers, and post-close information.

## XGBoost tuning

Three-fold `TimeSeriesSplit` evaluates four configurations within training only.

| Depth | Learning rate | Trees | CV R2 | CV MAE | CV MdAPE |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.10 | 300 | **0.8844** | **$168,641** | **9.60%** |
| 7 | 0.05 | 300 | 0.8764 | $177,301 | 10.34% |
| 4 | 0.10 | 300 | 0.8507 | $201,659 | 12.54% |
| 4 | 0.05 | 300 | 0.8287 | $217,972 | 13.56% |

The selected XGBoost configuration uses depth 7, learning rate 0.10, and 300 trees.

## Validation comparison

| Model | R2 | MAE | RMSE | MAPE | MdAPE |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8757 | **$171,103** | $334,436 | **12.72%** | **8.20%** |
| XGBoost | **0.8866** | $175,045 | **$319,472** | 13.82% | 9.67% |

XGBoost explains more overall variation and has lower RMSE, but Random Forest has lower MAE, MAPE, and MdAPE. Following the predefined validation rule, Random Forest is selected.

## June evaluation

| Model | Population | R2 | MAE | RMSE | MAPE | MdAPE |
|---|---|---:|---:|---:|---:|---:|
| Random Forest | Primary in-range | 0.8762 | **$170,135** | $335,700 | **12.77%** | **8.23%** |
| XGBoost | Primary in-range | **0.8906** | $174,249 | **$315,581** | 13.88% | 9.68% |
| Random Forest | Full June | 0.5999 | $231,306 | $955,058 | **22.72%** | **8.48%** |
| XGBoost | Full June | **0.6450** | **$229,604** | **$899,613** | 22.80% | 9.96% |

## Conclusion

Random Forest is the final model because it produces the lowest typical and average proportional error on validation and primary June. XGBoost is preferable only if the business objective prioritizes R2 and RMSE. The choice is therefore objective-dependent rather than evidence that one model dominates every metric.

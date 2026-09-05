# Week 4 - Linear Regression Baseline

**Notebook:** [`03_baseline_model.ipynb`](../03_baseline_model.ipynb)

## Objective

Establish a transparent Linear Regression baseline for California single-family `ClosePrice` prediction and compare it with a median-only naive benchmark.

## Evaluation design

| Split | Period | Primary rows |
|---|---|---:|
| Training | May 2025-April 2026 | 116,424 |
| Validation | May 2026 | 11,735 |
| Test | June 2026 | 12,543 |

The primary population uses `ClosePrice` and price-per-square-foot limits learned only from training data. All 12,827 June records are also evaluated as a robustness check. The model uses 63 leakage-safe, model-ready features; `ListPrice`, `OriginalListPrice`, target-derived fields, and post-close information are excluded.

## Results

| Model | Split | R2 | MAE | RMSE | MAPE | MdAPE |
|---|---|---:|---:|---:|---:|---:|
| Median Dummy | Validation | -0.1196 | $597,144 | $1,003,898 | 49.12% | 39.38% |
| Linear Regression | Validation | 0.6110 | $392,180 | $591,770 | 37.77% | 27.16% |
| Median Dummy | Test - primary in-range | -0.1138 | $592,401 | $1,006,734 | 48.89% | 39.38% |
| Linear Regression | Test - primary in-range | **0.6205** | **$384,250** | **$587,649** | **36.62%** | **26.97%** |
| Linear Regression | Full June robustness | 0.4814 | $450,399 | $1,087,423 | 45.58% | 27.61% |

## Interpretation

Linear Regression substantially outperforms the naive median benchmark and remains stable from validation to primary June test. Its 26.97% test MdAPE is nevertheless too high for a final pricing model. The lower full-June R2 and higher RMSE show that unusual and high-value transactions are difficult for a linear specification.

Large coefficients for location-frequency and geographic variables should be treated as model associations, not causal effects. Coefficient size also depends on scaling and encoding.

## Conclusion

The baseline establishes that the leakage-safe features contain meaningful signal, while motivating nonlinear models that can learn interactions between property size, geography, amenities, and market context.

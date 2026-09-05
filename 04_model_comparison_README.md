# Week 5 - Model Comparison

**Notebook:** [`04_model_comparison.ipynb`](../04_model_comparison.ipynb)

## Objective

Compare the Linear Regression baseline with Decision Tree and Random Forest models using the same 63 model-ready features, rows, chronological split, and evaluation metrics. A median dummy remains in the table as a naive benchmark.

## Model comparison

| Model | Validation R2 | Validation MdAPE | Primary June R2 | Primary June MAE | Primary June MdAPE |
|---|---:|---:|---:|---:|---:|
| Median Dummy | -0.1196 | 39.38% | -0.1138 | $592,401 | 39.38% |
| Linear Regression | 0.6110 | 27.16% | 0.6205 | $384,250 | 26.97% |
| Decision Tree | 0.8216 | 10.24% | 0.8220 | $203,541 | 10.20% |
| Random Forest | **0.8872** | **7.82%** | **0.8847** | **$164,265** | **8.01%** |

Random Forest is selected using validation MdAPE with MAE as the tie-breaker. It improves primary June R2 by 0.2642 and lowers MAE by $219,985 relative to Linear Regression.

## Full-June robustness

| Model | R2 | MAE | RMSE | MdAPE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.4814 | $450,399 | $1,087,423 | 27.61% |
| Decision Tree | 0.5752 | $263,258 | $984,177 | 10.48% |
| Random Forest | **0.6059** | **$225,137** | **$947,963** | **8.27%** |

The full-June R2 drop is driven by large squared errors on unusual or high-value transactions. MdAPE changes much less, which indicates that performance on a typical transaction remains comparatively stable.

## Feature importance

The strongest Random Forest signals are living area, longitude, latitude, bathrooms, ZIP-code frequency, county frequency, MLS-area frequency, and city frequency. Importance describes predictive usage inside this model and must not be interpreted causally.

## Conclusion

Random Forest clearly outperforms the linear and single-tree alternatives and becomes the benchmark for the feature-engineering and advanced-model stages.

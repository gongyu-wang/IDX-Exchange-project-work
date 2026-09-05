# Week 6 - Updated Linear Baseline

**Notebook:** [`03_baseline_model_updated.ipynb`](../03_baseline_model_updated.ipynb)

## Objective

Measure the incremental value of Week 6 feature engineering with the model family, chronological windows, evaluation population, and preprocessing rules held constant.

## Feature sets

The leakage-safe base pool contains 15 numeric, 10 categorical, and 7 binary fields. The updated set adds:

- bedroom-to-bathroom ratio;
- property age;
- living-area-to-lot-area ratio;
- amenity count;
- cyclical closing-month variables; and
- spatially assigned school district.

The school-district join matched 99.99% of training rows and 100% of validation and June rows. It identified 594 training districts, 455 validation districts, and 447 test districts.

## Controlled Linear Regression results

| Feature set | Split | Input fields | R2 | MAE | RMSE | MAPE | MdAPE |
|---|---|---:|---:|---:|---:|---:|---:|
| Old | Validation | 32 | 0.8451 | $226,063 | $373,359 | 21.07% | 14.94% |
| Updated | Validation | 39 | **0.8508** | **$219,890** | **$366,467** | **20.07%** | **14.33%** |
| Old | Primary June | 32 | 0.8465 | $225,013 | $373,730 | 20.88% | 15.19% |
| Updated | Primary June | 39 | **0.8525** | **$219,188** | **$366,304** | **20.01%** | **14.71%** |

The updated features improve primary June R2 by 0.0060, reduce MAE by $5,825, and reduce MdAPE by 0.48 percentage points.

## Full-June robustness

| Feature set | R2 | MAE | RMSE | MAPE | MdAPE |
|---|---:|---:|---:|---:|---:|
| Old | 0.6485 | $278,523 | $895,210 | 30.23% | 15.50% |
| Updated | **0.6538** | **$272,192** | **$888,392** | **29.04%** | **14.95%** |

## Interpretation

Feature engineering produces a modest but consistent improvement for Linear Regression. The result supports the new feature set, but the remaining 14.71% primary-test MdAPE shows that a linear model still cannot capture all nonlinear location and property interactions.

`FireplacesTotal` was entirely missing in the executed data and was skipped by the median imputer. This warning does not invalidate the run, but the field should be removed from the next feature definition unless future data supplies usable values.

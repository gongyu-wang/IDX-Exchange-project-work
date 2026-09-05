# Week 8 - Final Evaluation

**Notebook:** [`06_evaluation.ipynb`](../06_evaluation.ipynb)

## Final model

Random Forest is selected from May validation performance because it has lower MAE, MAPE, and MdAPE than tuned XGBoost. The fitted model uses 39 leakage-safe input fields.

## Overall performance

| Population | Rows | R2 | MAE | RMSE | MAPE | MdAPE | Within 10% | Within 20% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 11,735 | 0.8757 | $171,103 | $334,436 | 12.72% | 8.20% | 57.2% | 81.3% |
| June - primary in-range | 12,543 | **0.8762** | **$170,135** | $335,700 | 12.77% | 8.23% | 57.5% | 81.1% |
| Full June robustness | 12,827 | 0.5999 | $231,306 | $955,058 | 22.72% | 8.48% | 56.3% | 79.4% |

Validation-to-primary-test R2 changes by only +0.0004 and MdAPE by +0.03 percentage points. This indicates strong month-to-month stability for the intended in-range population. Full-June RMSE is much higher because squared-error metrics are dominated by a small number of unusual transactions.

## Performance by price band

| Price band | Rows | MAE | MAPE | MdAPE | Within 20% |
|---|---:|---:|---:|---:|---:|
| Under $500K | 1,634 | $58,118 | 15.69% | 8.55% | 74.7% |
| $500K-$750K | 2,679 | $65,822 | 10.66% | **6.05%** | 85.3% |
| $750K-$1M | 2,611 | $87,917 | **10.10%** | 6.50% | **87.9%** |
| $1M-$1.5M | 2,568 | $159,778 | 13.06% | 9.18% | 82.7% |
| $1.5M-$2M | 1,402 | $239,390 | 14.01% | 10.12% | 79.7% |
| $2M+ | 1,649 | $538,028 | 16.07% | 13.31% | 68.7% |

The model is strongest in the $500K-$1M range. Dollar and percentage errors increase for properties above $2M, which should receive closer manual review.

## Geographic findings

Among counties with at least 100 June observations, Merced has the lowest MdAPE at 5.44%, followed by Riverside at 6.03% and San Bernardino at 6.15%. Santa Cruz has the highest MdAPE at 20.76%, followed by San Mateo at 12.50%. Sparse counties and locally unusual properties require additional comparable-sales review.

## Error direction and review priority

- Underpredictions: 47.9%; median dollar error $71,992
- Overpredictions: 52.1%; median dollar error -$79,217
- Low review priority, within 10% error: 57.5%
- Moderate, 10%-20% error: 23.6%
- High, 20%-30% error: 9.6%
- Critical, above 30% error: 9.3%

The nearly balanced prediction direction indicates no large one-sided tendency at the aggregate level, although county and price-band bias should still be monitored.

## Business recommendation

Use the model as pricing decision support for typical California single-family properties, especially in the $500K-$1M range. Require manual comparable-sales review for luxury homes, sparse geographies, unusual property configurations, and cases flagged as high or critical risk. The prediction is not a certified appraisal.

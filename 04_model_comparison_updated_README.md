# Week 6 - Updated Model Comparison

**Notebook:** [`04_model_comparison_updated.ipynb`](../04_model_comparison_updated.ipynb)

## Objective

Complete the Week 6 feature-engineering evaluation by separating two questions:

1. Does the updated feature set improve the same Random Forest configuration?
2. Which model performs best after the updated feature set is applied?

## Old versus updated feature set

| Feature set | Validation R2 | Validation MdAPE | Primary June R2 | Primary June MdAPE |
|---|---:|---:|---:|---:|
| Old | **0.8789** | **8.04%** | **0.8811** | **8.09%** |
| Updated | 0.8745 | 8.23% | 0.8758 | 8.26% |

For Random Forest, the engineered feature set does not improve the controlled result: primary June R2 falls by 0.0053 and MdAPE increases by 0.17 percentage points. This is still an informative feature-engineering outcome. New features can help a linear model while adding redundancy or noise to a nonlinear model that already learns interactions from the base variables.

## Updated-feature model comparison

| Model | Validation R2 | Validation MdAPE | Primary June R2 | Primary June MAE | Primary June MdAPE |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 0.7469 | 19.72% | 0.7515 | $294,506 | 19.86% |
| Decision Tree | 0.8178 | 10.59% | 0.8108 | $210,077 | 10.31% |
| Random Forest | **0.8745** | **8.23%** | **0.8758** | **$170,932** | **8.26%** |

Random Forest remains the strongest updated-feature model. On complete June it records R2 0.6014, MAE $231,924, and MdAPE 8.51%, again showing that extreme transactions affect squared-error metrics much more than the typical proportional error.

## School-district layer

Spatial match coverage is effectively complete, so missing geography is not driving the result. District membership provides useful geographic context, but it should not be interpreted as school quality or a causal price effect.

## Conclusion

The feature-engineering hypothesis receives mixed evidence: the expanded features improve Linear Regression, but the controlled Random Forest performs slightly better with the old set. The correct conclusion is to retain feature engineering for interpretation and further testing, while validating whether each engineered variable earns a place in the final nonlinear model.

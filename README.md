# California Residential Property Close Price Prediction

This project develops and evaluates machine learning models for predicting the final sale price (`ClosePrice`) of single-family residential properties in California. It uses historical CRMLS sales records and follows a time-based validation strategy so that model performance is measured on later, unseen months.

## Project Objectives

- Prepare and encode CRMLS property data for machine learning.
- Establish a Linear Regression baseline.
- Compare the baseline with Decision Tree and Random Forest regressors.
- Engineer property-level and geographic features.
- Tune an XGBoost regressor and compare it with the earlier models.
- Evaluate performance with R-squared, MAE, MAPE, and MdAPE, including error analysis by price band.

## Dataset

The source data consists of monthly `CRMLSSold` files from the California Regional Multiple Listing Service (CRMLS), provided through IDX Exchange. The preprocessing notebook loads monthly files from January 2025 through May 2026 and restricts the analysis to:

```text
PropertyType = Residential
PropertySubType = SingleFamilyResidence
```

The prediction target is `ClosePrice`. Raw MLS files and prepared datasets are not included in this repository because they are externally provided and may be subject to access or licensing restrictions.

The final modeling notebooks use the following chronological split:

| Split | Period | Observations before model-specific column removal |
|---|---:|---:|
| Training | May 2025-April 2026 | 117,602 |
| Validation | May 2026 | 11,873 |
| Test | June 2026 | 12,685 |

Using later months for validation and testing provides a more realistic estimate of performance on future property sales than a random split.

## Data Preparation

The preprocessing and modeling workflow includes:

1. Concatenating monthly CRMLS sales files.
2. Filtering to residential single-family properties.
3. Converting inconsistent binary values such as `True`, `False`, `Fals`, and missing values into numeric indicators.
4. Converting date fields to datetime values.
5. One-hot encoding selected low-cardinality categorical variables.
6. Frequency encoding higher-cardinality location variables, including city, ZIP code, county, MLS area, and school district fields.
7. Removing model-excluded identifiers, text fields, target-derived fields, and unstable audit columns.
8. Dropping columns with more than 90% missingness in the training set.
9. Filling remaining numeric missing values with training-set means and applying the same values to validation and test data to avoid leakage.

## Feature Engineering

The updated Week 6 workflow adds the following features:

- **Bed/Bath Ratio:** bedrooms divided by bathrooms, with the bathroom denominator floored at one to prevent division by zero.
- **Property Age:** current year minus `YearBuilt`, with negative values set to zero.
- **Amenities Count:** sum of seven binary amenity indicators: view, waterfront, basement, private pool, attached garage, fireplace, and new construction.
- **School District:** property coordinates are spatially joined to California school district boundaries. `DistrictName` is then frequency encoded for modeling; unmatched properties are assigned `Unknown District`.

The geographic join requires `DistrictAreas.geojson`, derived from the California School District Areas 2024-25 boundary dataset.

> **Experiment note:** the original and updated notebooks use different validation/test months in addition to different features. Their results therefore represent successive workflow versions, not a controlled estimate of the isolated causal effect of feature engineering.

## Models Tested

| Model | Main configuration |
|---|---|
| Linear Regression | Scikit-learn default configuration |
| Decision Tree | `random_state=42` |
| Random Forest | 100 trees, `random_state=42`, all CPU cores |
| XGBoost | Manually tuned across tree depth, learning rate, and number of estimators |

The selected XGBoost configuration uses `max_depth=7`, `learning_rate=0.05`, `n_estimators=300`, and `random_state=42`.

## Final Results

Results below come from the final feature set and chronological split.

| Model | Validation R-squared | Test R-squared | Test MAE | Test MAPE | Test MdAPE |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 0.6359 | -1.3633 | $576,815.78 | 62.89% | 27.16% |
| Decision Tree | 0.7282 | 0.7178 | $256,101.42 | 18.98% | 11.96% |
| Random Forest | 0.8738 | 0.8639 | **$179,154.00** | **13.55%** | **8.35%** |
| XGBoost | **0.8785** | **0.8750** | $187,980.31 | 15.45% | 10.07% |

XGBoost achieved the highest validation and test R-squared values and showed similar performance across both sets, indicating good generalization. Random Forest produced the lowest MAE, MAPE, and MdAPE. Consequently, the preferred model depends on the business objective: XGBoost explains slightly more overall price variation, while Random Forest gives lower typical prediction errors.

The strongly negative test R-squared for Linear Regression indicates that its linear specification is not robust to the June 2026 test distribution and performs worse than predicting the test-set mean.

## XGBoost Performance by Price Band

| Actual close price | Test observations | MAE | MAPE | MdAPE |
|---|---:|---:|---:|---:|
| Under $500K | 1,742 | $85,410.85 | 23.46% | 12.98% |
| $500K-$750K | 2,744 | $87,796.51 | 14.05% | 8.42% |
| $750K-$1M | 2,579 | $115,203.57 | **13.11%** | 8.52% |
| $1M-$1.5M | 2,559 | $176,174.12 | 14.20% | 10.14% |
| $1.5M-$2M | 1,381 | $240,572.38 | 13.94% | 10.39% |
| $2M+ | 1,680 | $544,441.00 | 16.20% | 13.27% |

XGBoost has its lowest MAPE in the $750K-$1M segment. Percentage error is highest below $500K, while absolute error is largest above $2M.

## Repository Contents

| File | Purpose |
|---|---|
| `02_preprocessing_categorical_to_numerical.ipynb` | Loads monthly raw files, filters properties, converts categorical/binary fields, and prepares dates. |
| `03_baseline_model.ipynb` | Trains and evaluates the original Linear Regression baseline. |
| `03_baseline_model_updated.ipynb` | Re-runs the baseline after Week 6 feature engineering. |
| `04_model_comparison.ipynb` | Compares the original Linear Regression, Decision Tree, and Random Forest models. |
| `04_model_comparison_updated.ipynb` | Re-runs the three-model comparison with the updated feature set and time split. |
| `05_advanced_models.ipynb` | Trains and manually tunes XGBoost and compares all four models. |
| `06_evaluation.ipynb` | Reports final metrics and evaluates XGBoost errors by price band. |

## Setup

The notebooks were developed in Google Colab with data mounted from Google Drive. Python 3.10 or later is recommended.

Install the required packages:

```bash
python -m pip install jupyter pandas numpy scikit-learn xgboost geopandas shapely matplotlib seaborn
```

Before running the notebooks, provide:

- the monthly `CRMLSSoldYYYYMM.csv` source files;
- the prepared chronological train, validation, and test CSV files referenced by the modeling notebooks; and
- `DistrictAreas.geojson` for school-district spatial joins.

Update the hard-coded Colab paths, such as `/content/drive/My Drive/IDX Exchange/Data/`, to match your own environment.

## Reproducing the Analysis

Run the notebooks in the following order:

1. `01_exploration.ipynb`
2. `02_preprocessing_categorical_to_numerical.ipynb`
3. `03_baseline_model.ipynb`
4. `04_model_comparison.ipynb`
5. `03_baseline_model_updated.ipynb`
6. `04_model_comparison_updated.ipynb`
7. `05_advanced_models.ipynb`
8. `06_evaluation.ipynb`

For local Jupyter:

```bash
jupyter lab
```

Open each notebook, update its data paths, and select **Run All**. The advanced and evaluation notebooks repeat earlier preparation steps so they can be executed independently once the required prepared CSVs and geographic boundary file are available.

## Prediction Application

A Streamlit application was optional in Week 9 and is not included in the current project files. If an application is added later, it should load a saved preprocessing pipeline and trained model, apply the same feature transformations used during training, and return a predicted close price from user-provided property characteristics.

## Limitations and Next Steps

- Model performance may change over time as California housing conditions shift.
- Percentage errors are not uniform across price bands.
- School-district frequency is a useful geographic proxy but does not directly represent school quality or causal effects on price.
- Preprocessing is repeated across notebooks; packaging it into a fitted Scikit-learn pipeline would reduce inconsistency and leakage risk.
- A future controlled experiment should keep the same train/validation/test months when measuring the incremental contribution of new features.
- Saving the final model, preprocessing objects, and feature schema would support reproducible batch predictions or a Streamlit application.

## Tools and Libraries

Python, pandas, NumPy, scikit-learn, XGBoost, GeoPandas, Shapely, Matplotlib, Seaborn, Jupyter, and Google Colab.

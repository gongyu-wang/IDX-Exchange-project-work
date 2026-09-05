# Week 2 — Preprocessing

**Notebook:** [`02_preprocessing.ipynb`](../02_preprocessing.ipynb)

## Goal

Convert monthly raw CRMLS files into reproducible, model-ready chronological datasets while preventing information from later months from leaking into training.

## Final split

| Split | Period |
|---|---|
| Training | May 2025–April 2026 |
| Validation | May 2026 |
| Test | June 2026 |

The 12-month training window is unchanged; it rolls forward so that June 2026 can serve as the newest test month.

## Main tasks

- Discover available monthly files instead of hard-coding a data cutoff.
- Filter to residential single-family properties.
- Standardize inconsistent binary values and date fields.
- Remove identifiers, free text, target-derived fields, and unstable audit columns.
- Encode lower-cardinality categories and training-frequency encode selected geographic categories.
- Drop highly missing columns based on training data only.
- Fit missing-value rules on training data and reuse them for validation and test.
- Align the three datasets to a consistent feature schema.

## Leakage safeguards

- Missingness thresholds are learned from training only.
- Imputation values are calculated from training only.
- Frequency maps are learned from training only.
- Validation and test months are never used to fit preprocessing decisions.

## Expected outputs

- Full prepared training dataset; target-quantile rows are not removed only from training
- Prepared May 2026 validation dataset
- Prepared June 2026 test dataset
- Feature schema and quality checks
- Row counts and date-range verification

Downstream modeling notebooks learn `ClosePrice` and price-per-square-foot bounds from training only. They report the in-range population as the primary comparison and complete June as a robustness check.

## Next notebook

Use the prepared files in [`03_baseline_model.ipynb`](../03_baseline_model.ipynb).

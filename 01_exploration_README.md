# Week 1 — Data Exploration

**Notebook:** [`01_exploration.ipynb`](../01_exploration.ipynb)

## Goal

Understand the CRMLS sold-property data before building a modeling pipeline. This notebook establishes the unit of analysis, reviews coverage and data quality, and identifies fields that require special handling later.

## Main tasks

- Inspect dataset dimensions, column types, and representative records.
- Confirm the modeling target, `ClosePrice`.
- Review missingness and unusual values.
- Examine price distributions and key property characteristics.
- Check categorical cardinality and inconsistent binary labels.
- Identify IDs, text fields, target-derived fields, and audit fields that should not enter the model.

## Inputs

Monthly CRMLS sold-property files supplied through IDX Exchange. The analysis focuses on residential single-family transactions.

## Expected outputs

- Dataset overview and descriptive statistics
- Missing-value summary
- Target-distribution plots
- Notes on columns to clean, transform, or exclude
- Initial data-quality and leakage-risk observations

## Review questions

1. Does each row represent a closed property transaction?
2. Are price and date fields plausible and consistently typed?
3. Which fields would be unavailable at prediction time?
4. Which categorical columns need one-hot or frequency encoding?
5. Are there extreme observations that require investigation rather than automatic deletion?

## Limitations

Exploration describes the observed sample and does not establish causal relationships. Raw MLS coverage, reporting practices, and missingness may differ across locations and months.

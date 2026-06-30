# California Property Close Price Prediction

Machine learning prediction of California residential property sale prices using historical MLS transaction data.

## Overview

This project develops a machine learning model to predict the final sale price (ClosePrice) of residential single-family properties in California using historical Multiple Listing Service (MLS) transaction data.

The project is completed as part of a remote Data Science internship and follows a structured workflow including data exploration, preprocessing, feature engineering, model development, evaluation, and documentation.


## Project Objectives

The primary objectives of this project are:

- Explore and understand historical California real estate transaction data.
- Preprocess and clean the dataset for machine learning.
- Build regression models to predict property closing prices.
- Compare multiple machine learning algorithms.
- Evaluate model performance using standard regression metrics.
- Document the complete data science workflow for reproducibility.


## Dataset

The dataset consists of historical California residential property transactions provided through the CRMLS (California Regional Multiple Listing Service).

Only properties satisfying the following conditions are included:

- **PropertyType**: Residential
- **PropertySubType**: SingleFamilyResidence

The target variable is:

- **ClosePrice** – Final property sale price

Example input features include:

- Living Area
- Bedrooms
- Bathrooms
- Lot Size
- Year Built
- Geographic Information
- Additional engineered features

**Note**: The dataset is proprietary and is not included in this repository.


# How to Run
**1. Clone the repository**
git clone https://github.com/your-username/california-property-price-prediction.git
cd california-property-price-prediction

**2. Install dependencies**
pip install -r requirements.txt

**3. Prepare the dataset**

Download the CRMLS datasets and place them into the appropriate data directory:

data/raw/

**4. Run the notebooks**

Execute the notebooks in order:

1. 01_exploration.ipynb
2. 02_preprocessing.ipynb
3. 03_baseline_model.ipynb
4. 04_model_comparison.ipynb
5. 05_advanced_models.ipynb
6. 06_evaluation.ipynb


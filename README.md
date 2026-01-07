# Household Energy_Consumption end to end Portfolio
This repository dedicated for end-to-end Household energy consumption predictor, a regression problem to predict household energy consumption based on  usage behavior

# Streamlit Deploy
[streamlit app](https://householdelectricityconsumptionpredhary.streamlit.app/)

# Project Overview

- Dataset: [Household Consumption Data](https://www.kaggle.com/datasets/samxsam/household-energy-consumption)
- Objective: Help households understand and estimate their daily electricity use so they can make more informed, sustainable decisions.  
- Approach: Train a regression model on a household energy dataset and expose it through a Streamlit UI where users can input features (e.g., household size, temperature, air conditioning, peak‑hour usage) and obtain a kWh prediction.  
- Tech stack: Python, scikit‑learn, pandas, Streamlit, joblib/pickle for model persistence.

# Features

- User‑friendly web form:
  - Household size (number of people)  
  - Average daily temperature (°C)  
  - Peak‑hours consumption (kWh)  
  - Binary options such as "Has Air Conditioning"  
- Machine learning:
  - Preprocessing pipeline for numeric and categorical features  
  - Regressor model trained and stored as `best_model.sav`  
-  Deployment‑ready:
  - Simple `app.py` Streamlit script  
  - `requirements.txt` for streamlit setup

# File Structure

```
.
├── End_to_End_Household_Energy_Consumption.ipynb  # Notebook with full EDA + modeling pipeline
├── Household_Energy_Comp.csv                     # Raw dataset used for training
├── best_model.sav                                # Serialized trained model (regressor)
├── app.py                                        # Streamlit application
├── requirements.txt                              # Python dependencies
└── README.md                                     # Project documentation
```

# Prerequisites
- Python 3.8+ (used Python 3.13.9 Kernel)
- pip or conda package manager

## Running Jupyter Notebook
It is advised to use VS Code to run the code.

# Model deployment using streamlit
- Step 1: Open python or conda terminal
- Step 2: Change into desired virtual environment
- Step 3: Change directory to desired ML folder
- Step 4: Run Streamlit as shown below
```
streamlit run app.py
```

# Model Performance
- Best tuned XGBoost parameter:
  - subsample: 0.7
  - estimators: 100
  - max depth: 6
  - learning rate: 0.1
  - colsample_bytree: 0.8
- The tuned XGBoost model achieved an impressive R squared of score of 0.999, indicating near-perfect fit to the training data
- Its Mean Absolute Percentage Error (MAPE) is just 0.098%, showcasing exceptional prediction accuracy relative to actual values.
- Root Mean Square Error (RMSE) stands at 0.237, reflecting minimal average prediction errors in kWh units.

# Required Libraries
```
- Matplotlib 3.0.2
- Numpy 2.3.5
- Pandas 2.3.3
- Sklearn (ScikitLearn) 1.7.2
- Shap 0.50.0
- Seaborn 0.13.2
- catboost 1.2.8
- lightgbm 4.6.0
- xgboost 3.1.2
- pickle (built-in)
```

# ⚡ Calgary Household Energy Consumption — End-to-End ML Portfolio

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?logo=streamlit) ![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle) ![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?logo=tensorflow) ![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning project that predicts and forecasts **daily household electricity consumption (kWh)** for 140 synthetic Calgary, Canada households over 2 years (2024–2025). The project covers the full ML pipeline: exploratory data analysis, feature engineering, regression baseline modelling, hyperparameter tuning, and LSTM time series forecasting — all deployed via a live two-tab Streamlit web app.

---

## 🌐 Live Demo

> 🚀 [**Launch the Streamlit App →**](https://householdelectricityconsumption.streamlit.app/)
> 📊 [**View the Dataset on Kaggle →**](https://www.kaggle.com/datasets/yonathanhary/household-energy-consumption-synthetic/data)

---

## 📁 Project Structure

```
📦 Household_Energy_Consumption_end_to_end_Portfolio
├── 📓 Calgary_Household_Energy_EDA_Regression.ipynb    # Notebook 1: EDA + Regression
├── 📓 Calgary_Household_Energy_LSTM_Forecasting.ipynb  # Notebook 2: LSTM Time Series Forecasting
├── 🌐 app.py                                           # Streamlit app (2 tabs: XGBoost + LSTM)
├── 🤖 best_model.sav                                   # Trained XGBoost model (GridSearchCV tuned)
├── 🧠 lstm_best.keras                                  # Best LSTM checkpoint (ModelCheckpoint)
├── 🧠 lstm_model.keras                                 # Final exported LSTM model
├── 💾 lstm_scaler.pkl                                  # MinMaxScaler for all features
├── 💾 lstm_target_scaler.pkl                           # MinMaxScaler for target (Daily_kWh)
├── 🗃️ calgary_household_energy_synthetic.csv           # Synthetic dataset (102,340 rows)
├── 📋 requirements.txt                                 # Python dependencies
└── 📖 README.md
```

> 📊 The dataset is also publicly available on [Kaggle](https://www.kaggle.com/datasets/yonathanhary/household-energy-consumption-synthetic/data).

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| Source | Synthetically generated (see methodology) |
| Households | 140 distinct IDs |
| Date Range | 2024-01-01 → 2025-12-31 (daily) |
| Total Records | ~102,340 rows |
| Target Variable | `Daily_kWh` |
| Location Context | Calgary, Alberta, Canada |

### Features

| Column | Description |
|---|---|
| `Household_ID` | Unique household identifier |
| `Date` | Calendar date |
| `Outside_Temperature_C` | Simulated daily temperature based on Calgary historical averages |
| `Household_Size` | Number of occupants (2–6) |
| `Living_Area_m2` | Floor area in m² (25–150) |
| `Has_EV_Car` | Binary — 1 if household owns an EV |
| `Max_AC_Hours` | Maximum daily AC capacity (0–10 hrs) |
| `AC_Hours_Used` | Actual AC hours used |
| `Tariff_Rate_CAD_kWh` | Calgary ENMAX tariff in CAD $/kWh |
| `Daily_Cost_CAD` | Estimated daily electricity cost |
| `Daily_kWh` | **Target — total daily energy usage** |

---

## 📓 Notebook 1 — EDA & Regression

**File:** `Calgary_Household_Energy_EDA_Regression.ipynb`

### Part 1 — Exploratory Data Analysis (EDA)
- Dataset shape, types, missing values
- Univariate distributions (histograms, boxplots)
- Correlation heatmap
- Seasonal consumption patterns (monthly/quarterly)
- EV vs non-EV consumption comparison
- Temperature vs kWh scatter analysis
- Household size & living area segmentation

### Part 2 — Feature Engineering
- Extract `Month`, `DayOfWeek`, `IsWeekend`, `Season` from `Date`
- Heating Degree Day (HDD) and Cooling Degree Day (CDD) features
- Monthly sunlight hours mapping (Calgary)

### Part 3 — Regression Modelling

**5 Baseline Models:**
| # | Model | Notes |
|---|---|---|
| 1 | Linear Regression | Interpretable baseline |
| 2 | Ridge Regression | L2 regularization |
| 3 | Decision Tree | Non-linear, no scaling needed |
| 4 | Random Forest | Ensemble, handles interactions |
| 5 | XGBoost | Gradient boosting baseline |

**Tuned Model:** XGBoost with `GridSearchCV` (5-fold CV)

---

## 📓 Notebook 2 — LSTM Time Series Forecasting

**File:** `Calgary_Household_Energy_LSTM_Forecasting.ipynb`

Forecasts the next N days of daily kWh for a given household using a **multivariate many-to-one LSTM** with a 30-day lookback window.

### Sections
1. **Time Series EDA** — Aggregate trend, monthly boxplot, autocorrelation (ACF)
2. **Sequence Building** — Sliding window sequences, MinMaxScaler, 70/15/15 split
3. **LSTM Architecture** — `LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(1)`
4. **Training** — EarlyStopping, ReduceLROnPlateau, ModelCheckpoint callbacks
5. **Evaluation** — RMSE, MAE, R² on test set; Actual vs Predicted plot; Residual analysis
6. **Multi-Step Forecast** — Recursive 30-day ahead forecast with ±12% confidence band
7. **Model Export** — Saves `lstm_model.keras`, `lstm_scaler.pkl`, `lstm_target_scaler.pkl`

### LSTM Input Features
```
Daily_kWh, Outside_Temperature_C, Has_EV_Car,
Household_Size, Living_Area_m2, AC_Hours_Used
```

---

## 🖥️ Streamlit App

The app has **two tabs**, deployed at [householdelectricityconsumption.streamlit.app](https://householdelectricityconsumption.streamlit.app/):

| Tab | Model | Description |
|---|---|---|
| 🧠 XGBoost — Daily Prediction | XGBoost (GridSearchCV) | Input household profile + weather → predict a single day's kWh |
| 📈 LSTM — Time Series Forecast | LSTM (TensorFlow/Keras) | Select a household → generate N-day ahead forecast with chart & table |

**Tab 1 Outputs:** Predicted kWh · Daily & monthly cost in CAD · Consumption tier (🟢 Low / 🟡 Medium / 🔴 High)

**Tab 2 Outputs:** Forecast chart (raw + 7-day smoothed + ±12% band) · Total forecast kWh & cost · Expandable forecast table

### Running Locally

```bash
git clone https://github.com/YonathanHH/Household_Energy_Consumption_end_to_end_Portfolio.git
cd Household_Energy_Consumption_end_to_end_Portfolio
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Methodology — Dataset Generation

The synthetic dataset is physically motivated, not random noise. Daily kWh is the sum of:

1. **Base load** — household size + floor area (appliances, hot water)
2. **Heating load** — degrees below 18°C comfort baseline × area (Calgary winters are severe)
3. **Cooling load** — AC hours × 0.9 kWh/hr
4. **EV charging** — 8–14 kWh/session, ~65% of days for EV households
5. **Lighting load** — inversely proportional to monthly sunlight hours
6. **Weekend effect** — +10% on weekends
7. **Lifestyle multiplier** — per-household fixed noise (±15%) for behavioural diversity

**Temperature** is sampled from `Normal(monthly_mean, monthly_std)` using Calgary historical climate data. **Tariff** reflects real ENMAX billing: monthly variable RRO in 2024, fixed ROLR 12.06 ¢/kWh in 2025.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Regression | scikit-learn, xgboost |
| Tuning | GridSearchCV (5-fold CV) |
| Forecasting | TensorFlow / Keras (LSTM) |
| App | Streamlit (2-tab layout) |
| Dataset hosting | Kaggle |
| Version control | GitHub |

---

## 📈 Results

### Notebook 1 — Regression Model Comparison

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | TBD | TBD | TBD |
| Ridge | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| XGBoost (baseline) | TBD | TBD | TBD |
| **XGBoost (tuned — GridSearchCV)** | **3.786** | **2.572** | **0.7148** |

### 🔑 Top Feature Importances (Tuned XGBoost)

| Rank | Feature | Interpretation |
|---|---|---|
| 🥇 1 | `Has_EV_Car` | EV charging dominates daily consumption (~8–14 kWh/session) |
| 🥈 2 | `Sunlight_Hours` | Fewer daylight hours → more artificial lighting load |
| 🥉 3 | `Household_Size` | More occupants → higher base + lighting load |

> **Note:** The R² of 0.71 reflects realistic prediction difficulty — EV charging days introduce stochastic variance (~65% charging probability) that is intentionally non-deterministic in the dataset generation.

### Notebook 2 — LSTM Forecasting

| Metric | Value |
|---|---|
| RMSE | TBD |
| MAE | TBD |
| R² | TBD |
| Lookback Window | 30 days |
| Forecast Horizon | Up to 90 days (configurable in app) |
| Architecture | `LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(1)` |

---

## 👤 Author

**Yonathan Hary Hutagalung**
MSc Sustainable Energy Science — Reykjavik University
BSc Geology — University of Canterbury

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/yonathanhary)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?logo=kaggle)](https://www.kaggle.com/yonathanhary)

---

## 📄 License

This project is licensed under the **MIT License**.

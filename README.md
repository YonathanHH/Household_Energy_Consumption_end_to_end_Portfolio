# ⚡ Calgary Household Energy Consumption — End-to-End ML Portfolio

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit) ![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle) ![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning project that predicts **daily household electricity consumption (kWh)** for 140 synthetic Calgary, Canada households over 2 years (2024–2025). The project covers the full ML pipeline: data exploration, feature engineering, baseline modelling, hyperparameter tuning, and a deployed Streamlit web app.

---

## 🌐 Live Demo

> 🚀 [**Launch the Streamlit App →**](https://your-streamlit-url.streamlit.app)  
> 📊 [**View the Dataset on Kaggle →**](https://www.kaggle.com/datasets/yonathanhary/household-energy-consumption-synthetic/data)

---

## 📁 Project Structure

```
📦 Household_Energy_Consumption_end_to_end_Portfolio
├── 📓 Calgary_Household_Energy_EDA_Regression.ipynb  # Full notebook: EDA + Modelling
├── 🌐 app.py                                         # Streamlit prediction app
├── 📋 requirements.txt                               # Python dependencies
└── 📖 README.md
```

> **Note:** The dataset (`calgary_household_energy_synthetic.csv`) is hosted on Kaggle.  
> The notebook downloads it automatically via the Kaggle API.

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

## 🔬 Notebook Outline

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
- Encode categorical columns

### Part 3 — Regression Modelling

**5 Baseline Models:**
| # | Model | Notes |
|---|---|---|
| 1 | Linear Regression | Interpretable baseline |
| 2 | Ridge Regression | L2 regularization |
| 3 | Decision Tree | Non-linear, no scaling needed |
| 4 | Random Forest | Ensemble, handles interactions |
| 5 | XGBoost | Gradient boosting baseline |

**Tuned Model:**
- XGBoost with `GridSearchCV` (5-fold CV)
- Metrics: RMSE, MAE, R²

---

## 🖥️ Streamlit App

The app lets users input household parameters and receive a real-time predicted daily energy consumption.

**Inputs:**
- Outside temperature (°C)
- Household size (2–6 people)
- Living area (m²)
- EV ownership
- AC hours used
- Day of week / Is weekend

**Outputs:**
- Predicted `Daily_kWh`
- Estimated daily & monthly cost in CAD
- Consumption tier label (Low / Medium / High)

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
| Modelling | scikit-learn, xgboost |
| Tuning | GridSearchCV |
| App | Streamlit |
| Dataset hosting | Kaggle |
| Version control | GitHub |

---

## 📈 Results

> Results will be updated after notebook execution.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | TBD | TBD | TBD |
| Ridge | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| XGBoost (baseline) | TBD | TBD | TBD |
| **XGBoost (tuned)** | **TBD** | **TBD** | **TBD** |

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

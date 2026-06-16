# app.py — Calgary Household Energy Consumption Predictor
# Dataset: https://www.kaggle.com/datasets/yonathanhary/household-energy-consumption-synthetic

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="⚡ Calgary Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 2rem; }
    .prediction-card {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        color: white; padding: 2rem; border-radius: 16px;
        text-align: center; margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(26,115,232,0.3);
    }
    .metric-card {
        background: white; border-radius: 12px; padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
    }
    .tier-low    { color: #2e7d32; font-weight: bold; }
    .tier-medium { color: #f57c00; font-weight: bold; }
    .tier-high   { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall",   10: "Fall",  11: "Fall"}

MONTH_SUNLIGHT = {1:5.6,2:7.0,3:8.8,4:10.2,5:11.2,6:12.3,
                  7:13.0,8:11.4,9:9.0,10:7.4,11:5.4,12:4.9}

# Calgary 2025 ROLR tariff (ENMAX)
TARIFF_CAD = 0.1206

def get_tier(kwh):
    if kwh < 12:
        return "🟢 Low", "tier-low"
    elif kwh < 25:
        return "🟡 Medium", "tier-medium"
    else:
        return "🔴 High", "tier-high"

@st.cache_resource
def load_model():
    model_path = "best_model.sav"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Calgary_Logo.svg/320px-Calgary_Logo.svg.png", width=160)
    st.markdown("### ⚡ About this App")
    st.markdown("""
    Predicts **daily household electricity consumption (kWh)** for Calgary, Canada households.

    - 📊 **Dataset**: [Kaggle — Calgary Synthetic Energy](https://www.kaggle.com/datasets/yonathanhary/household-energy-consumption-synthetic)
    - 🧠 **Model**: XGBoost (GridSearchCV tuned)
    - 📅 **Data Range**: Jan 2024 – Dec 2025
    """)
    st.markdown("---")
    st.markdown("Built by **Yonathan Hary Hutagalung**")
    st.markdown("MSc Sustainable Energy Science")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ Calgary Household Energy Consumption Predictor")
st.markdown("*Enter your household details to predict daily electricity usage in kWh*")
st.markdown("---")

# ── Input Form ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 🏠 Household Profile")
    household_size = st.slider("Household Size (people)", min_value=2, max_value=6, value=3, step=1)
    living_area    = st.slider("Living Area (m²)", min_value=25, max_value=150, value=80, step=5)
    has_ev         = st.toggle("🔋 Has EV Car?", value=False)
    max_ac_hours   = st.slider("Max AC Capacity (hours/day)", min_value=0, max_value=10, value=4, step=1)

with col_right:
    st.markdown("### 🌡️ Weather & Usage")
    temperature    = st.slider("Outside Temperature (°C)", min_value=-30.0, max_value=35.0, value=5.0, step=0.5,
                               help="Calgary range: ~-28°C (winter) to ~+24°C (summer)")
    ac_hours_used  = st.slider("AC Hours Used Today", min_value=0.0, max_value=float(max_ac_hours), value=0.0, step=0.5,
                               disabled=(max_ac_hours == 0))
    month          = st.selectbox("Month", options=list(range(1, 13)),
                                  format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                          "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
    is_weekend     = st.toggle("📅 Is Weekend?", value=False)

# ── Derived Features ─────────────────────────────────────────────────────────
hdd = max(0, 18 - temperature)
cdd = max(0, temperature - 18)
season = SEASON_MAP[month]
season_enc = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}[season]
sunlight = MONTH_SUNLIGHT[month]

feature_df = pd.DataFrame([{
    "Outside_Temperature_C": temperature,
    "Household_Size":        household_size,
    "Living_Area_m2":        living_area,
    "Has_EV_Car":            int(has_ev),
    "Max_AC_Hours":          max_ac_hours,
    "AC_Hours_Used":         ac_hours_used,
    "Tariff_Rate_CAD_kWh":   TARIFF_CAD,
    "Month":                 month,
    "DayOfWeek":             5 if is_weekend else 2,
    "IsWeekend":             int(is_weekend),
    "Season_enc":            season_enc,
    "HDD":                   hdd,
    "CDD":                   cdd,
    "Sunlight_Hours":        sunlight,
}])

# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("⚡ Predict Daily Consumption", use_container_width=True, type="primary")

if predict_btn:
    if model is None:
        st.warning("""
        ⚠️ **No trained model found** (`best_model.sav` missing).

        Please run the notebook `Calgary_Household_Energy_EDA_Regression.ipynb` first to train
        and export the model, then add `best_model.sav` to this directory.
        """)
    else:
        try:
            prediction = model.predict(feature_df)[0]
            prediction = max(1.5, float(prediction))

            daily_cost  = prediction * TARIFF_CAD
            monthly_est = daily_cost * 30
            tier_label, tier_cls = get_tier(prediction)

            # Main result card
            st.markdown(f"""
            <div class="prediction-card">
                <div style="font-size:1.1rem; opacity:0.85;">Predicted Daily Consumption</div>
                <div style="font-size:3.5rem; font-weight:900; margin:0.3rem 0;">{prediction:.2f} kWh</div>
                <div style="font-size:1rem; opacity:0.75;">Calgary · {season} · {'Weekend' if is_weekend else 'Weekday'}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("💰 Est. Daily Cost",  f"CAD ${daily_cost:.2f}")
            m2.metric("📅 Monthly Estimate",  f"CAD ${monthly_est:.0f}")
            m3.metric("🌡️ Temperature",       f"{temperature:.1f} °C")
            m4.metric("🔥 Heating Degrees",   f"{hdd:.1f} HDD")

            st.markdown(f"**Consumption Tier:** <span class='{tier_cls}'>{tier_label}</span>",
                        unsafe_allow_html=True)

            # Feature summary expander
            with st.expander("🔍 View Input Features Sent to Model"):
                st.dataframe(feature_df.T.rename(columns={0: "Value"}), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; font-size:0.85rem;">
⚡ Calgary Household Energy Predictor &nbsp;|&nbsp;
Dataset: <a href="https://www.kaggle.com/datasets/yonathanhary/household-energy-consumption-synthetic" target="_blank">Kaggle</a> &nbsp;|&nbsp;
GitHub: <a href="https://github.com/YonathanHH/Household_Energy_Consumption_end_to_end_Portfolio" target="_blank">YonathanHH</a><br>
<small>Model trained on 102,340 synthetic daily records · Calgary, Alberta · 2024–2025</small>
</div>
""", unsafe_allow_html=True)

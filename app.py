# app.py — Calgary Household Energy Consumption Predictor
# Dataset: https://www.kaggle.com/datasets/yonathanhary/household-energy-consumption-synthetic

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    .forecast-card {
        background: linear-gradient(135deg, #e65100 0%, #bf360c 100%);
        color: white; padding: 2rem; border-radius: 16px;
        text-align: center; margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(230,81,0,0.3);
    }
    .tier-low    { color: #2e7d32; font-weight: bold; }
    .tier-medium { color: #f57c00; font-weight: bold; }
    .tier-high   { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
SEASON_MAP = {12:"Winter",1:"Winter",2:"Winter",
              3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",
              9:"Fall",10:"Fall",11:"Fall"}
MONTH_SUNLIGHT = {1:5.6,2:7.0,3:8.8,4:10.2,5:11.2,6:12.3,
                  7:13.0,8:11.4,9:9.0,10:7.4,11:5.4,12:4.9}
TARIFF_CAD = 0.1206
LOOKBACK   = 30
LSTM_FEATURES = ['Daily_kWh','Outside_Temperature_C','Has_EV_Car',
                 'Household_Size','Living_Area_m2','AC_Hours_Used']

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_tier(kwh):
    if kwh < 12:   return "🟢 Low",    "tier-low"
    elif kwh < 25: return "🟡 Medium", "tier-medium"
    else:          return "🔴 High",   "tier-high"

# ── Model Loaders ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_xgb_model():
    if os.path.exists("best_model.sav"):
        with open("best_model.sav", "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_lstm_artifacts():
    try:
        from tensorflow.keras.models import load_model as keras_load
        model         = keras_load("lstm_best.keras")
        scaler        = joblib.load("lstm_scaler.pkl")
        target_scaler = joblib.load("lstm_target_scaler.pkl")
        return model, scaler, target_scaler
    except Exception:
        return None, None, None

@st.cache_data
def load_dataset():
    if os.path.exists("calgary_household_energy_synthetic.csv"):
        df = pd.read_csv("calgary_household_energy_synthetic.csv", parse_dates=["Date"])
        return df.sort_values(["Household_ID","Date"]).reset_index(drop=True)
    return None

xgb_model                        = load_xgb_model()
lstm_model, scaler, tgt_scaler   = load_lstm_artifacts()
df_data                          = load_dataset()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Calgary_Logo.svg/320px-Calgary_Logo.svg.png", width=160)
    st.markdown("### ⚡ About this App")
    st.markdown("""
    Predict and forecast **daily household electricity (kWh)** for Calgary, Canada.

    - 🧠 **Tab 1**: XGBoost single-day prediction
    - 📈 **Tab 2**: LSTM multi-day time series forecast
    - 📊 **Dataset**: [Kaggle](https://www.kaggle.com/datasets/yonathanhary/household-energy-consumption-synthetic)
    - 📅 **Data range**: Jan 2024 – Dec 2025
    """)
    st.markdown("---")
    st.markdown("Built by **Yonathan Hary Hutagalung**")
    st.markdown("MSc Sustainable Energy Science")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ Calgary Household Energy — Prediction & Forecasting")
st.markdown("*XGBoost daily prediction · LSTM time series forecasting · Calgary, Alberta*")
st.markdown("---")

# ── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧠 XGBoost — Daily Prediction", "📈 LSTM — Time Series Forecast"])


# ================================================================
# TAB 1 — XGBoost Daily Prediction
# ================================================================
with tab1:
    st.markdown("### 🏠 Enter Household Details")
    st.markdown("Predict the **next single day** energy consumption based on household profile and weather.")
    st.markdown("")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**Household Profile**")
        household_size = st.slider("Household Size (people)", min_value=2, max_value=6, value=3, step=1, key="xgb_size")
        living_area    = st.slider("Living Area (m²)", min_value=25, max_value=150, value=80, step=5, key="xgb_area")
        has_ev         = st.toggle("🔋 Has EV Car?", value=False, key="xgb_ev")
        max_ac_hours   = st.slider("Max AC Capacity (hrs/day)", min_value=0, max_value=10, value=4, step=1, key="xgb_ac")

    with col_right:
        st.markdown("**Weather & Date**")
        temperature  = st.slider("Outside Temperature (°C)", min_value=-30.0, max_value=35.0, value=5.0, step=0.5, key="xgb_temp")
        ac_hours_used = st.slider("AC Hours Used Today", min_value=0.0, max_value=float(max_ac_hours), value=0.0, step=0.5,
                                  disabled=(max_ac_hours == 0), key="xgb_ac_used")
        month        = st.selectbox("Month", options=list(range(1, 13)), key="xgb_month",
                                    format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                            "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
        is_weekend   = st.toggle("📅 Is Weekend?", value=False, key="xgb_weekend")

    # Derived
    hdd        = max(0, 18 - temperature)
    cdd        = max(0, temperature - 18)
    season     = SEASON_MAP[month]
    season_enc = {"Winter":0,"Spring":1,"Summer":2,"Fall":3}[season]
    sunlight   = MONTH_SUNLIGHT[month]

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

    st.markdown("---")
    if st.button("⚡ Predict Daily Consumption", use_container_width=True, type="primary", key="xgb_btn"):
        if xgb_model is None:
            st.warning("⚠️ `best_model.sav` not found. Run Notebook 1 first to export the model.")
        else:
            try:
                prediction  = max(1.5, float(xgb_model.predict(feature_df)[0]))
                daily_cost  = prediction * TARIFF_CAD
                monthly_est = daily_cost * 30
                tier_label, tier_cls = get_tier(prediction)

                st.markdown(f"""
                <div class="prediction-card">
                    <div style="font-size:1.1rem;opacity:0.85;">Predicted Daily Consumption</div>
                    <div style="font-size:3.5rem;font-weight:900;margin:0.3rem 0;">{prediction:.2f} kWh</div>
                    <div style="font-size:1rem;opacity:0.75;">{season} · {'Weekend' if is_weekend else 'Weekday'} · {temperature:.1f}°C</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("💰 Daily Cost",     f"CAD ${daily_cost:.2f}")
                m2.metric("📅 Monthly Est.",   f"CAD ${monthly_est:.0f}")
                m3.metric("🌡️ Temperature",    f"{temperature:.1f} °C")
                m4.metric("🔥 Heating Degrees", f"{hdd:.1f} HDD")
                st.markdown(f"**Consumption Tier:** <span class='{tier_cls}'>{tier_label}</span>", unsafe_allow_html=True)

                with st.expander("🔍 Input features sent to model"):
                    st.dataframe(feature_df.T.rename(columns={0:"Value"}), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")


# ================================================================
# TAB 2 — LSTM Time Series Forecast
# ================================================================
with tab2:
    st.markdown("### 📈 LSTM Time Series Forecast")
    st.markdown("""
    Select a household from the dataset and generate a **multi-day ahead forecast** using
    the trained LSTM model (`LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(1)`).
    The model uses the last **30 days** of actual data as the seed window.
    """)
    st.markdown("")

    if lstm_model is None or scaler is None or tgt_scaler is None:
        st.warning("⚠️ LSTM model files not found (`lstm_best.keras`, `lstm_scaler.pkl`, `lstm_target_scaler.pkl`). Run Notebook 2 first.")
    elif df_data is None:
        st.warning("⚠️ Dataset CSV not found. Please add `calgary_household_energy_synthetic.csv` to the app directory.")
    else:
        # ── Controls
        ctrl_col1, ctrl_col2 = st.columns([1, 1], gap="large")

        with ctrl_col1:
            households   = sorted(df_data['Household_ID'].unique().tolist())
            selected_hh  = st.selectbox("🏠 Select Household", options=households, index=0, key="lstm_hh")

        with ctrl_col2:
            forecast_days = st.slider("📅 Forecast Horizon (days)", min_value=7, max_value=90, value=30, step=7, key="lstm_days")

        st.markdown("---")

        if st.button("📈 Run LSTM Forecast", use_container_width=True, type="primary", key="lstm_btn"):
            try:
                # ── Prepare household series
                df_hh = df_data[df_data['Household_ID'] == selected_hh][LSTM_FEATURES + ['Date']].copy()
                df_hh = df_hh.set_index('Date').sort_index()

                if len(df_hh) < LOOKBACK:
                    st.error(f"Not enough data for {selected_hh}. Need at least {LOOKBACK} days.")
                    st.stop()

                # ── Scale
                series_vals = df_hh[LSTM_FEATURES].values
                scaled      = scaler.transform(series_vals)

                # ── Recursive forecast
                last_window         = scaled[-LOOKBACK:].copy()
                future_preds_scaled = []

                for _ in range(forecast_days):
                    x_in   = last_window.reshape(1, LOOKBACK, len(LSTM_FEATURES))
                    pred_s = lstm_model.predict(x_in, verbose=0)[0, 0]
                    future_preds_scaled.append(pred_s)
                    new_row    = last_window[-1].copy()
                    new_row[0] = pred_s
                    last_window = np.vstack([last_window[1:], new_row])

                # ── Inverse transform
                future_preds = tgt_scaler.inverse_transform(
                    np.array(future_preds_scaled).reshape(-1, 1)
                ).flatten()
                future_preds = np.maximum(future_preds, 1.5)

                last_date    = df_hh.index[-1]
                future_dates = pd.date_range(
                    last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D'
                )

                # ── Header metrics
                avg_forecast = future_preds.mean()
                total_kwh    = future_preds.sum()
                total_cost   = total_kwh * TARIFF_CAD

                st.markdown(f"""
                <div class="forecast-card">
                    <div style="font-size:1.1rem;opacity:0.85;">LSTM Forecast — {selected_hh} · Next {forecast_days} Days</div>
                    <div style="font-size:3rem;font-weight:900;margin:0.3rem 0;">{avg_forecast:.2f} kWh/day avg</div>
                    <div style="font-size:1rem;opacity:0.75;">
                        {future_dates[0].strftime('%b %d, %Y')} → {future_dates[-1].strftime('%b %d, %Y')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                f1, f2, f3 = st.columns(3)
                f1.metric("⚡ Total Forecast kWh",   f"{total_kwh:.1f} kWh")
                f2.metric("💰 Total Est. Cost",      f"CAD ${total_cost:.2f}")
                f3.metric("📅 Forecast Period",      f"{forecast_days} days")

                # ── Forecast plot
                HIST_DAYS = min(60, len(df_hh))
                hist_dates = df_hh.index[-HIST_DAYS:]
                hist_vals  = df_hh['Daily_kWh'].values[-HIST_DAYS:]
                smooth     = pd.Series(future_preds, index=future_dates).rolling(7, min_periods=1, center=True).mean()

                fig, ax = plt.subplots(figsize=(14, 4))
                ax.plot(hist_dates,   hist_vals,             color='steelblue',  lw=1.3, label='Historical (actual)')
                ax.plot(future_dates, future_preds,          color='darkorange', lw=0.8, alpha=0.45, label='Daily forecast (raw)')
                ax.plot(future_dates, smooth.values,         color='crimson',    lw=2.0, label='7-day rolling mean')
                ax.fill_between(future_dates,
                                future_preds * 0.88, future_preds * 1.12,
                                alpha=0.12, color='darkorange', label='±12% band')
                ax.axvline(last_date, color='gray', lw=1.2, ls=':', label='Forecast start')
                ax.set_title(f'LSTM {forecast_days}-Day Forecast — {selected_hh}', fontsize=12, fontweight='bold')
                ax.set_ylabel('Daily kWh')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                plt.xticks(rotation=30)
                ax.legend(fontsize=9); ax.grid(alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # ── Forecast table
                with st.expander("📊 View Full Forecast Table"):
                    forecast_df = pd.DataFrame({
                        'Date':          future_dates.strftime('%Y-%m-%d'),
                        'Predicted_kWh': future_preds.round(3),
                        'Est_Cost_CAD':  (future_preds * TARIFF_CAD).round(4),
                    })
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Forecast error: {e}")
                st.info("Make sure `lstm_best.keras`, `lstm_scaler.pkl`, and `lstm_target_scaler.pkl` are all present in the app directory.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:gray;font-size:0.85rem;">
⚡ Calgary Household Energy &nbsp;|&nbsp;
Dataset: <a href="https://www.kaggle.com/datasets/yonathanhary/household-energy-consumption-synthetic" target="_blank">Kaggle</a> &nbsp;|&nbsp;
GitHub: <a href="https://github.com/YonathanHH/Household_Energy_Consumption_end_to_end_Portfolio" target="_blank">YonathanHH</a><br>
<small>XGBoost regression · LSTM time series forecasting · Calgary, Alberta · 2024–2025</small>
</div>
""", unsafe_allow_html=True)

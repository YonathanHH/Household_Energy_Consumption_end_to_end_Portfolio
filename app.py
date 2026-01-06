# app.py - Streamlit Energy Consumption Predictor
# Install: pip install streamlit scikit-learn pandas numpy

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="⚡ Energy Consumption Predictor",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .stContainer {
        background: white;
        border-radius: 10px;
        padding: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("## ⚡ Household Energy Consumption Predictor")
st.markdown("*Predict your daily energy consumption in kWh*")

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('best_model.sav', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'best_model.sav' not found!")
        st.info("Please upload your pickle file to the app directory.")
        return None

model = load_model()

if model is not None:
    # Input form with 4 features
    st.markdown("### Enter Your Household Details:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        household_size = st.number_input(
            "Household Size (people)",
            min_value=1,
            max_value=20,
            value=4,
            step=1,
            help="Number of people living in the household"
        )
        
        avg_temperature = st.slider(
            "Average Temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=25.0,
            step=0.5,
            help="Average daily temperature in Celsius"
        )
    
    with col2:
        has_ac_option = st.radio(
            "Has Air Conditioning?",
            options=["Yes", "No"],
            index=0,
            help="Does the household have AC?"
        )
        has_ac = "Yes" if has_ac_option == "Yes" else "No"
        
        peak_hours_usage = st.number_input(
            "Peak Hours Usage (kWh)",
            min_value=0.0,
            max_value=20.0,
            value=3.5,
            step=0.5,
            help="Energy consumed during peak hours (typically 6pm-10pm)"
        )
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Household_Size': [household_size],
        'Avg_Temperature_C': [avg_temperature],
        'Has_AC': [has_ac],
        'Peak_Hours_Usage_kWh': [peak_hours_usage]
    })
    
    # Make prediction
    if st.button("Predict Consumption", use_container_width=True, type="primary"):
        try:
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.markdown(
                f"""
                <div class="prediction-box">
                    <div style="font-size: 1.2rem; opacity: 0.9;">Predicted Daily Consumption</div>
                    <div class="prediction-value">{prediction:.2f} kWh</div>
                    <div style="font-size: 1rem; margin-top: 1rem; opacity: 0.8;">
                        Based on your household details
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Additional insights
            st.markdown("### 📊 Insights:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Household Size",
                    f"{household_size} people",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Temperature",
                    f"{avg_temperature:.1f}°C",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Peak Usage",
                    f"{peak_hours_usage:.1f} kWh",
                    delta=None
                )
            
            # Cost estimation (example: $0.12 per kWh)
            cost_per_kwh = 0.12
            estimated_cost = prediction * cost_per_kwh
            
            st.info(f"💰 **Estimated Daily Cost:** ${estimated_cost:.2f} (at ${cost_per_kwh}/kWh)")
            st.success(f"📅 **Monthly Estimate:** ${estimated_cost * 30:.2f}")
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.9rem;">
        ⚡ Built with ML | Powered by Streamlit<br>
        <small>Model: XGBoost with Log Transformation</small>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.stop()

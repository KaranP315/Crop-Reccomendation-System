"""
streamlit_app.py — Streamlit frontend for the Crop Recommendation System
==========================================================================
Provides a user-friendly UI with sliders/inputs for soil & weather data
and displays the predicted best crop.

Usage:
    streamlit run frontend/streamlit_app.py
"""

import streamlit as st
import requests

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="centered",
)

# ──────────────────────────────────────────────
# Custom CSS for a polished look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stButton > button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        font-size: 18px;
        padding: 0.6rem;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1b5e20;
        color: white;
    }
    .crop-result {
        font-size: 28px;
        font-weight: 700;
        color: #1b5e20;
        text-align: center;
        padding: 1rem;
        border: 2px solid #2e7d32;
        border-radius: 12px;
        background: #e8f5e9;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("🌾 Crop Recommendation System")
st.markdown(
    "Enter soil nutrient levels and weather conditions below, "
    "and the model will recommend the **best crop** to grow."
)

# ──────────────────────────────────────────────
# Backend URL (Flask API)
# ──────────────────────────────────────────────
API_URL = "http://127.0.0.1:5000/predict"

# ──────────────────────────────────────────────
# Input form
# ──────────────────────────────────────────────
st.subheader("📝 Enter Soil & Weather Data")

col1, col2 = st.columns(2)

with col1:
    nitrogen    = st.slider("Nitrogen (N)",       min_value=0,   max_value=140, value=50,  step=1)
    phosphorus  = st.slider("Phosphorus (P)",     min_value=5,   max_value=145, value=50,  step=1)
    potassium   = st.slider("Potassium (K)",      min_value=5,   max_value=205, value=50,  step=1)
    temperature = st.slider("Temperature (°C)",   min_value=0.0, max_value=50.0, value=25.0, step=0.1)

with col2:
    humidity    = st.slider("Humidity (%)",        min_value=10.0, max_value=100.0, value=70.0, step=0.1)
    ph          = st.slider("pH Level",            min_value=0.0,  max_value=14.0,  value=6.5,  step=0.1)
    rainfall    = st.slider("Rainfall (mm)",       min_value=20.0, max_value=300.0, value=100.0, step=0.5)

# ──────────────────────────────────────────────
# Predict button
# ──────────────────────────────────────────────
if st.button("🌱 Predict Best Crop"):
    payload = {
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            crop = result.get("recommended_crop", "Unknown")
            st.markdown(
                f'<div class="crop-result">🌿 Recommended Crop: <strong>{crop.upper()}</strong></div>',
                unsafe_allow_html=True,
            )
            st.balloons()
        else:
            error = response.json().get("error", "Unknown error")
            st.error(f"❌ API Error: {error}")

    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Could not connect to the backend API. "
            "Make sure the Flask server is running:\n\n"
            "```\npython backend/app.py\n```"
        )
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Flask & scikit-learn")

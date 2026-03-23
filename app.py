import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("model.pkl")

# Page config
st.set_page_config(page_title="Electricity Cost Predictor")
st.title("⚡ Electricity Cost Prediction App")

st.write("Fill in the building/environment details below to predict electricity cost:")

# Input fields
site_area = st.number_input("Site Area (sq ft)", min_value=300,max_value=6000)
water_consumption = st.number_input("Water Consumption", min_value=600,max_value=13000)
recycling_rate = st.slider("Recycling Rate (%)", min_value=0, max_value=100)
utilisation_rate = st.slider("Utilisation Rate (%)", min_value=0, max_value=100)
air_quality_index = st.number_input("Air Quality Index", min_value=0,max_value=200)
issue_resolution_time = st.number_input("Issue Resolution Time (in days)", min_value=1,max_value=200)

# User-friendly structure type options
structure_display = st.selectbox(
    "Structure Type",
    options=["Mixed use", "Commercial", "Residential", "Industrial"]
)

# Map structure type to encoded values
structure_map = {
    "Mixed use": 0,
    "Commercial": 1,
    "Residential": 2,
    "Industrial": 3
}
structure_type_encoded = structure_map[structure_display]

resident_count = st.number_input("Resident Count", format="%.6f")

# Prediction button
if st.button("Predict Electricity Cost"):
    features = np.array([[
        site_area, water_consumption, recycling_rate, utilisation_rate,
        air_quality_index, issue_resolution_time, structure_type_encoded, resident_count
    ]])

    # Ensure XGBoost-compatible dtype
    features = features.astype(np.float32)

    # Predict
    prediction = model.predict(features)
    st.success(f"Predicted Electricity Cost: ₹{prediction[0]:,.2f}")

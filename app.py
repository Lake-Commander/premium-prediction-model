import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load model
model = joblib.load('random_forest_model.pkl')

# App title
st.title("🧮 Insurance Premium Predictor")
st.markdown("Enter client details below to estimate the insurance premium.")

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
    children = st.slider("Number of Children", 0, 5, 1)
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    submitted = st.form_submit_button("Predict")

# Preprocess input
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "sex": 1 if gender == "Male" else 0,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker == "Yes" else 0,
        "region_northeast": 1 if region == "northeast" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }])

    # Reorder columns to match model input
    expected_columns = model.feature_names_in_
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # Check and apply scaler if available
    scaler_path = "models/scaler.pkl"  # Update if your path differs
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"💰 Estimated Premium: **₦{prediction:,.2f}**")

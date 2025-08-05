# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load model, scaler, and feature order
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_order.pkl")

st.title("Insurance Premium Prediction")

st.markdown("Enter customer details to estimate their insurance premium.")

# === USER INPUT FORM ===
with st.form("input_form"):
    age = st.slider("Age", 18, 100, 30)
    health_score = st.slider("Health Score", 0, 100, 80)
    credit_score = st.slider("Credit Score", 0, 100, 70)
    vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
    annual_income = st.number_input("Annual Income ($)", min_value=1000.0, max_value=1_000_000.0, value=50_000.0)
    insurance_duration = st.slider("Insurance Duration (years)", 1, 30, 5)
    num_dependents = st.slider("Number of Dependents", 0, 10, 2)
    previous_claims = st.number_input("Previous Claims ($)", min_value=0.0, max_value=1_000_000.0, value=0.0)

    # Categorical
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    property_type = st.selectbox("Property Type", ["House", "Condo", "Apartment"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    customer_feedback = st.selectbox("Customer Feedback", ["Good", "Average", "Poor"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed", "Unknown"])
    education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    exercise_frequency = st.selectbox("Exercise Frequency", ["Rarely", "Monthly", "Weekly", "Daily"])

    submitted = st.form_submit_button("Predict")

# === FEATURE ENGINEERING & PREDICTION ===
if submitted:
    annual_income_log = np.log1p(annual_income)
    previous_claims_log = np.log1p(previous_claims)

    # Base numerical features
    base_features = {
        "Health Score": health_score,
        "Age": age,
        "Credit Score": credit_score,
        "Vehicle Age": vehicle_age,
        "Annual Income": annual_income,
        "Annual Income_log": annual_income_log,
        "Insurance Duration": insurance_duration,
        "Number of Dependents": num_dependents,
        "Previous Claims": previous_claims,
        "Previous Claims_log": previous_claims_log,
    }

    # One-hot categorical features
    one_hot_features = {
        f"Gender_{gender}": 1,
        f"Smoking Status_{smoking_status}": 1,
        f"Location_{location}": 1,
        f"Property Type_{property_type}": 1,
        f"Policy Type_{policy_type}": 1,
        f"Customer Feedback_{customer_feedback}": 1,
        f"Marital Status_{marital_status}": 1,
        f"Occupation_{occupation}": 1,
        f"Education Level_{education_level}": 1,
        f"Exercise Frequency_{exercise_frequency}": 1,
    }

    # Combine and fill missing features with 0
    all_features = {**base_features, **one_hot_features}
    final_input = pd.DataFrame([[all_features.get(feat, 0) for feat in feature_order]], columns=feature_order)

    # Scale input
    scaled_input = scaler.transform(final_input)

    # Predict log premium and transform back
    pred_log = model.predict(scaled_input)[0]
    premium = np.expm1(pred_log)

    st.success(f"Estimated Insurance Premium: **${premium:,.2f}**")

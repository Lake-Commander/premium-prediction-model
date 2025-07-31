# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("xgboost_(tuned).pkl")

# Define full feature list as used in training
FEATURES = [
    'Health Score', 'Age', 'Credit Score', 'Vehicle Age', 'Annual Income_log', 'Annual Income',
    'Insurance Duration', 'Number of Dependents', 'Previous Claims', 'Previous Claims_log',
    'Gender_Male', 'Smoking Status_Yes', 'Location_Suburban', 'Property Type_Condo', 'Location_Urban',
    'Policy Type_Premium', 'Customer Feedback_Poor', 'Marital Status_Single', 'Property Type_House',
    'Occupation_Unknown', 'Marital Status_Married', 'Exercise Frequency_Monthly',
    'Exercise Frequency_Rarely', 'Education Level_PhD', 'Customer Feedback_Good',
    'Policy Type_Comprehensive', "Education Level_Master's", 'Exercise Frequency_Weekly',
    'Education Level_High School'
]

st.title("🚗 Insurance Premium Prediction App")

# Collect user inputs
with st.form("predict_form"):
    st.subheader("Enter Customer Details")

    health_score = st.slider("Health Score", 1, 100, 50)
    age = st.slider("Age", 18, 80, 30)
    credit_score = st.slider("Credit Score", 300, 850, 600)
    vehicle_age = st.number_input("Vehicle Age (years)", 0, 30, 5)
    annual_income = st.number_input("Annual Income ($)", 1000, 1_000_000, 50_000)
    insurance_duration = st.slider("Insurance Duration (years)", 0, 30, 3)
    dependents = st.slider("Number of Dependents", 0, 10, 1)
    previous_claims = st.number_input("Previous Claims ($)", 0.0, 1_000_000.0, 0.0)

    gender = st.selectbox("Gender", ["Female", "Male"])
    smoker = st.selectbox("Smoking Status", ["No", "Yes"])
    location = st.selectbox("Location", ["Rural", "Suburban", "Urban"])
    property_type = st.selectbox("Property Type", ["Apartment", "Condo", "House"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    feedback = st.selectbox("Customer Feedback", ["Neutral", "Good", "Poor"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    occupation = st.selectbox("Occupation", ["Known", "Unknown"])
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    exercise = st.selectbox("Exercise Frequency", ["Never", "Rarely", "Monthly", "Weekly", "Daily"])

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Feature Engineering
        data = {
            'Health Score': health_score,
            'Age': age,
            'Credit Score': credit_score,
            'Vehicle Age': vehicle_age,
            'Annual Income_log': np.log1p(annual_income),
            'Annual Income': annual_income,
            'Insurance Duration': insurance_duration,
            'Number of Dependents': dependents,
            'Previous Claims': previous_claims,
            'Previous Claims_log': np.log1p(previous_claims),
        }

        # One-hot encoding manually for selected options
        one_hot = {
            'Gender_Male': 1 if gender == "Male" else 0,
            'Smoking Status_Yes': 1 if smoker == "Yes" else 0,
            'Location_Suburban': 1 if location == "Suburban" else 0,
            'Location_Urban': 1 if location == "Urban" else 0,
            'Property Type_Condo': 1 if property_type == "Condo" else 0,
            'Property Type_House': 1 if property_type == "House" else 0,
            'Policy Type_Premium': 1 if policy_type == "Premium" else 0,
            'Policy Type_Comprehensive': 1 if policy_type == "Comprehensive" else 0,
            'Customer Feedback_Poor': 1 if feedback == "Poor" else 0,
            'Customer Feedback_Good': 1 if feedback == "Good" else 0,
            'Marital Status_Single': 1 if marital_status == "Single" else 0,
            'Marital Status_Married': 1 if marital_status == "Married" else 0,
            'Occupation_Unknown': 1 if occupation == "Unknown" else 0,
            'Exercise Frequency_Monthly': 1 if exercise == "Monthly" else 0,
            'Exercise Frequency_Rarely': 1 if exercise == "Rarely" else 0,
            'Exercise Frequency_Weekly': 1 if exercise == "Weekly" else 0,
            'Education Level_PhD': 1 if education == "PhD" else 0,
            "Education Level_Master's": 1 if education == "Master's" else 0,
            'Education Level_High School': 1 if education == "High School" else 0,
        }

        # Merge all inputs
        full_input = {**data, **one_hot}

        # Ensure all required features are present
        for col in FEATURES:
            full_input.setdefault(col, 0)

        # Order columns correctly
        input_df = pd.DataFrame([full_input])[FEATURES]

        # Make prediction
        log_pred = model.predict(input_df)[0]
        premium = np.expm1(log_pred)

        st.success(f"💰 Predicted Insurance Premium: **${premium:,.2f}**")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math

# Load model
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("🧮 Insurance Premium Predictor")
st.markdown("Enter client details below to estimate the insurance premium.")

with st.form("prediction_form"):
    age = st.slider("Age", 18, 64, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    annual_income = st.slider("Annual Income (₦)", 0, 150000, 50000)
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    dependents = st.slider("Number of Dependents", 0, 4, 1)
    education = st.selectbox("Education Level", ["Master's", "Bachelor's", "PhD", "High School"])
    occupation = st.selectbox("Occupation", ["Self-Employed", "Employed", "Unemployed", "Unknown"])
    health_score = st.slider("Health Score", 0.0, 100.0, 50.0)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Comprehensive", "Premium", "Basic"])
    previous_claims = st.slider("Previous Claims", 0, 9, 0)
    vehicle_age = st.slider("Vehicle Age (years)", 0, 19, 5)
    credit_score = st.slider("Credit Score", 300, 850, 600)
    insurance_duration = st.slider("Insurance Duration (years)", 1, 9, 3)
    smoking = st.selectbox("Smoking Status", ["Yes", "No"])
    exercise = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"])
    property_type = st.selectbox("Property Type", ["Condo", "House", "Apartment"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Feature construction
    input_features = {
        'Premium Amount_log': 0,  # dummy; not used in prediction
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

        # Binary flags for one-hot encoded features
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Smoking Status_Yes': 1 if smoking == 'Yes' else 0,
        'Location_Suburban': 1 if location == 'Suburban' else 0,
        'Property Type_Condo': 1 if property_type == 'Condo' else 0,
        'Location_Urban': 1 if location == 'Urban' else 0,
        'Policy Type_Premium': 1 if policy_type == 'Premium' else 0,
        'Customer Feedback_Poor': 1 if feedback == 'Poor' else 0,
        'Marital Status_Single': 1 if marital_status == 'Single' else 0,
        'Property Type_House': 1 if property_type == 'House' else 0,
        'Occupation_Unknown': 1 if occupation == 'Unknown' else 0,
        'Marital Status_Married': 1 if marital_status == 'Married' else 0,
        'Exercise Frequency_Monthly': 1 if exercise == 'Monthly' else 0,
        'Exercise Frequency_Rarely': 1 if exercise == 'Rarely' else 0,
        'Education Level_PhD': 1 if education == 'PhD' else 0,
        'Customer Feedback_Good': 1 if feedback == 'Good' else 0,
        'Policy Type_Comprehensive': 1 if policy_type == 'Comprehensive' else 0,
        "Education Level_Master's": 1 if education == "Master's" else 0,
        'Exercise Frequency_Weekly': 1 if exercise == 'Weekly' else 0,
        'Education Level_High School': 1 if education == 'High School' else 0,

        'Premium Amount': 0  # dummy placeholder; not needed for prediction
    }

    # Convert to DataFrame in the right order
    expected_order = list(input_features.keys())
    input_df = pd.DataFrame([input_features])[expected_order]

    # Run prediction
    prediction_log = model.predict(input_df)[0]
    prediction_naira = np.expm1(prediction_log)  # reverse log1p

    # Display result
    st.success(f"💰 Estimated Premium: **₦{prediction_naira:,.2f}**")

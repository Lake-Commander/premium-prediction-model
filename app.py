import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("🧮 Insurance Premium Predictor")

st.markdown("Enter client details below to estimate the insurance premium.")

with st.form("prediction_form"):
    age = st.slider("Age", 18, 64, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    annual_income = st.slider("Annual Income", 0, 150000, 50000)
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    dependents = st.slider("Number of Dependents", 0, 4, 1)
    education = st.selectbox("Education Level", ["Master's", "Bachelor's", "PhD", "High School"])
    occupation = st.selectbox("Occupation", ["Self-Employed", "Employed", "Unemployed"])
    health_score = st.slider("Health Score", 0.0, 100.0, 50.0)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Comprehensive", "Premium", "Basic"])
    previous_claims = st.slider("Previous Claims", 0, 9, 0)
    vehicle_age = st.slider("Vehicle Age", 0, 19, 5)
    credit_score = st.slider("Credit Score", 300, 850, 600)
    insurance_duration = st.slider("Insurance Duration (years)", 1, 9, 3)
    smoking = st.selectbox("Smoking Status", ["Yes", "No"])
    exercise = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"])
    property_type = st.selectbox("Property Type", ["Condo", "House", "Apartment"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Construct DataFrame manually in correct order (update with your actual model features)
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Annual Income": annual_income,
        "Marital Status": marital_status,
        "Number of Dependents": dependents,
        "Education Level": education,
        "Occupation": occupation,
        "Health Score": health_score,
        "Location": location,
        "Policy Type": policy_type,
        "Previous Claims": previous_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": insurance_duration,
        "Smoking Status": smoking,
        "Exercise Frequency": exercise,
        "Customer Feedback": feedback,
        "Property Type": property_type
    }])

    # You may need to apply the same preprocessing steps here as in training
    # (encoding, scaling, etc.)

    # NOTE: Apply preprocessing here if needed (e.g., OneHotEncoder, scaler)
    # For demo, assuming model accepts raw input

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Premium: **₦{prediction:,.2f}**")

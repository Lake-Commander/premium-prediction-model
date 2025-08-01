import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load("models/random_forest_model.pkl")

# Define feature columns in correct order
feature_columns = [
    'Health Score', 'Age', 'Credit Score', 'Vehicle Age', 'Annual Income_log',
    'Insurance Duration', 'Number of Dependents', 'Previous Claims_log',
    'Gender_Male', 'Smoking Status_Yes', 'Location_Suburban', 'Property Type_Condo',
    'Location_Urban', 'Policy Type_Premium', 'Customer Feedback_Poor',
    'Marital Status_Single', 'Property Type_House', 'Occupation_Unknown',
    'Marital Status_Married', 'Exercise Frequency'
]

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

st.title("💰 Insurance Premium Prediction App")

# User Inputs
st.header("Enter Customer Information:")

health_score = st.slider("Health Score", 0, 100, 70)
age = st.slider("Age", 18, 100, 30)
credit_score = st.slider("Credit Score", 300, 850, 600)
vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
annual_income = st.number_input("Annual Income (₦)", min_value=100000.0, value=500000.0)
insurance_duration = st.slider("Insurance Duration (Years)", 1, 30, 5)
num_dependents = st.slider("Number of Dependents", 0, 10, 2)
previous_claims = st.number_input("Previous Claim Amount (₦)", min_value=0.0, value=0.0)

gender = st.selectbox("Gender", ["Male", "Female"])
smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
property_type = st.selectbox("Property Type", ["House", "Condo", "Apartment"])
policy_type = st.selectbox("Policy Type", ["Premium", "Basic", "Standard"])
customer_feedback = st.selectbox("Customer Feedback", ["Excellent", "Average", "Poor"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
occupation = st.selectbox("Occupation", ["Unknown", "Employed", "Self-Employed"])
exercise_freq = st.slider("Exercise Frequency (per week)", 0, 7, 3)

# Feature Engineering
input_data = {
    'Health Score': health_score,
    'Age': age,
    'Credit Score': credit_score,
    'Vehicle Age': vehicle_age,
    'Annual Income_log': np.log1p(annual_income),
    'Insurance Duration': insurance_duration,
    'Number of Dependents': num_dependents,
    'Previous Claims_log': np.log1p(previous_claims),
    'Gender_Male': 1 if gender == 'Male' else 0,
    'Smoking Status_Yes': 1 if smoking_status == 'Yes' else 0,
    'Location_Suburban': 1 if location == 'Suburban' else 0,
    'Location_Urban': 1 if location == 'Urban' else 0,
    'Property Type_Condo': 1 if property_type == 'Condo' else 0,
    'Property Type_House': 1 if property_type == 'House' else 0,
    'Policy Type_Premium': 1 if policy_type == 'Premium' else 0,
    'Customer Feedback_Poor': 1 if customer_feedback == 'Poor' else 0,
    'Marital Status_Single': 1 if marital_status == 'Single' else 0,
    'Marital Status_Married': 1 if marital_status == 'Married' else 0,
    'Occupation_Unknown': 1 if occupation == 'Unknown' else 0,
    'Exercise Frequency': exercise_freq
}

# Ensure all features are accounted for
for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0

# Convert to DataFrame
input_df = pd.DataFrame([input_data])[feature_columns]

# Predict
if st.button("Predict Premium"):
    prediction_log = model.predict(input_df)[0]
    prediction = np.expm1(prediction_log)
    formatted = f"₦{prediction:,.2f}"
    st.success(f"✅ Estimated Premium: {formatted}")

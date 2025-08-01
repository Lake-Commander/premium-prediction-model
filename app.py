import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model and scaler ===
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_order = joblib.load('models/feature_order.pkl')

st.title("Insurance Premium Prediction")
st.header("Enter Customer Information")

# === User Inputs ===
health_score = st.slider("Health Score", 0.0, 1.0, 0.5)
age = st.number_input("Age", 18, 100, 30)
credit_score = st.slider("Credit Score", 0.0, 1.0, 0.5)
vehicle_age = st.number_input("Vehicle Age (years)", 0, 20, 2)
annual_income = st.number_input("Annual Income (₦)", 10000.0, 10000000.0, 500000.0)
insurance_duration = st.number_input("Insurance Duration (years)", 0, 50, 5)
num_dependents = st.number_input("Number of Dependents", 0, 10, 2)
previous_claims = st.number_input("Previous Claims", 0, 10, 1)

# === Categoricals with encoding ===
gender = st.radio("Gender", ["Male", "Female"])
smoking = st.radio("Smoking Status", ["Yes", "No"])
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
property_type = st.selectbox("Property Type", ["House", "Condo", "Apartment"])
policy_type = st.selectbox("Policy Type", ["Basic", "Premium", "Comprehensive"])
customer_feedback = st.selectbox("Customer Feedback", ["Good", "Average", "Poor"])
marital_status = st.radio("Marital Status", ["Single", "Married"])
occupation = st.selectbox("Occupation", ["Employed", "Unemployed", "Unknown"])
exercise_freq = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

# === Build the input feature dict ===
input_dict = {
    'Health Score': health_score,
    'Age': age,
    'Credit Score': credit_score,
    'Vehicle Age': vehicle_age,
    'Annual Income': annual_income,
    'Annual Income_log': np.log1p(annual_income),
    'Insurance Duration': insurance_duration,
    'Number of Dependents': num_dependents,
    'Previous Claims': previous_claims,
    'Previous Claims_log': np.log1p(previous_claims),
    'Gender_Male': 1 if gender == "Male" else 0,
    'Smoking Status_Yes': 1 if smoking == "Yes" else 0,
    'Location_Urban': 1 if location == "Urban" else 0,
    'Location_Suburban': 1 if location == "Suburban" else 0,
    'Property Type_House': 1 if property_type == "House" else 0,
    'Property Type_Condo': 1 if property_type == "Condo" else 0,
    'Policy Type_Premium': 1 if policy_type == "Premium" else 0,
    'Policy Type_Comprehensive': 1 if policy_type == "Comprehensive" else 0,
    'Customer Feedback_Poor': 1 if customer_feedback == "Poor" else 0,
    'Customer Feedback_Good': 1 if customer_feedback == "Good" else 0,
    'Marital Status_Single': 1 if marital_status == "Single" else 0,
    'Marital Status_Married': 1 if marital_status == "Married" else 0,
    'Occupation_Unknown': 1 if occupation == "Unknown" else 0,
    'Exercise Frequency_Monthly': 1 if exercise_freq == "Monthly" else 0,
    'Exercise Frequency_Rarely': 1 if exercise_freq == "Rarely" else 0,
    'Exercise Frequency_Weekly': 1 if exercise_freq == "Weekly" else 0,
    'Education Level_PhD': 1 if education == "PhD" else 0,
    'Education Level_Master\'s': 1 if education == "Master's" else 0,
    'Education Level_High School': 1 if education == "High School" else 0
}

# Fill missing dummy vars with 0
for col in feature_order:
    if col not in input_dict:
        input_dict[col] = 0

# Convert to DataFrame and reorder
input_df = pd.DataFrame([input_dict])[feature_order]

# === Predict ===
if st.button("Predict Premium"):
    scaled_input = scaler.transform(input_df)
    log_prediction = model.predict(scaled_input)
    prediction = np.expm1(log_prediction[0])
    st.success(f"Estimated Premium: ₦{prediction:,.2f}")

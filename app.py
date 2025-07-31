# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

# Expected feature columns (from training, excluding target)
expected_features = [
    'Health Score', 'Age', 'Credit Score', 'Vehicle Age', 'Annual Income_log',
    'Annual Income', 'Insurance Duration', 'Number of Dependents',
    'Previous Claims', 'Previous Claims_log', 'Gender_Male', 'Smoking Status_Yes',
    'Location_Suburban', 'Property Type_Condo', 'Location_Urban',
    'Policy Type_Premium', 'Customer Feedback_Poor', 'Marital Status_Single',
    'Property Type_House', 'Occupation_Unknown', 'Marital Status_Married',
    'Exercise Frequency_Monthly', 'Exercise Frequency_Rarely', 'Education Level_PhD',
    'Customer Feedback_Good', 'Policy Type_Comprehensive',
    "Education Level_Master's", 'Exercise Frequency_Weekly',
    'Education Level_High School'
]

# Title
st.title("Insurance Premium Prediction")

st.markdown("Enter customer details below to predict the insurance premium.")

# User Inputs
health_score = st.slider("Health Score", 0, 100, 50)
age = st.slider("Age", 18, 100, 30)
credit_score = st.slider("Credit Score", 300, 900, 600)
vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
annual_income = st.number_input("Annual Income", min_value=1000.0, step=1000.0)
insurance_duration = st.slider("Insurance Duration (years)", 0, 30, 5)
num_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
previous_claims = st.number_input("Previous Claims", min_value=0, step=1)

# Categorical inputs
gender = st.selectbox("Gender", ["Female", "Male"])
smoking = st.selectbox("Smoking Status", ["No", "Yes"])
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
property_type = st.selectbox("Property Type", ["House", "Condo", "Apartment"])
policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
customer_feedback = st.selectbox("Customer Feedback", ["Good", "Average", "Poor"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
occupation = st.selectbox("Occupation", ["Professional", "Unemployed", "Unknown", "Other"])
exercise_freq = st.selectbox("Exercise Frequency", ["Rarely", "Monthly", "Weekly", "Daily"])
education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

# Log-transformed features
annual_income_log = np.log1p(annual_income)
previous_claims_log = np.log1p(previous_claims)

# Initialize feature dictionary with 0s
input_data = dict.fromkeys(expected_features, 0)

# Fill numeric fields
input_data['Health Score'] = health_score
input_data['Age'] = age
input_data['Credit Score'] = credit_score
input_data['Vehicle Age'] = vehicle_age
input_data['Annual Income'] = annual_income
input_data['Annual Income_log'] = annual_income_log
input_data['Insurance Duration'] = insurance_duration
input_data['Number of Dependents'] = num_dependents
input_data['Previous Claims'] = previous_claims
input_data['Previous Claims_log'] = previous_claims_log

# Fill one-hot encoded categories
if gender == "Male":
    input_data['Gender_Male'] = 1
if smoking == "Yes":
    input_data['Smoking Status_Yes'] = 1
if location == "Urban":
    input_data['Location_Urban'] = 1
elif location == "Suburban":
    input_data['Location_Suburban'] = 1
if property_type == "Condo":
    input_data['Property Type_Condo'] = 1
elif property_type == "House":
    input_data['Property Type_House'] = 1
if policy_type == "Premium":
    input_data['Policy Type_Premium'] = 1
elif policy_type == "Comprehensive":
    input_data['Policy Type_Comprehensive'] = 1
if customer_feedback == "Poor":
    input_data['Customer Feedback_Poor'] = 1
elif customer_feedback == "Good":
    input_data['Customer Feedback_Good'] = 1
if marital_status == "Single":
    input_data['Marital Status_Single'] = 1
elif marital_status == "Married":
    input_data['Marital Status_Married'] = 1
if occupation == "Unknown":
    input_data['Occupation_Unknown'] = 1
if exercise_freq == "Monthly":
    input_data['Exercise Frequency_Monthly'] = 1
elif exercise_freq == "Rarely":
    input_data['Exercise Frequency_Rarely'] = 1
elif exercise_freq == "Weekly":
    input_data['Exercise Frequency_Weekly'] = 1
if education_level == "PhD":
    input_data['Education Level_PhD'] = 1
elif education_level == "Master's":
    input_data["Education Level_Master's"] = 1
elif education_level == "High School":
    input_data['Education Level_High School'] = 1

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Insurance Premium Amount: ₦{prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

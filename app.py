import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define the correct feature order used in training (excluding target)
feature_order = [
    "Health Score", "Age", "Credit Score", "Vehicle Age", "Insurance Duration", "Annual Income",
    "Annual Income_log", "Previous Claims_log", "Previous Claims", "Number of Dependents",
    "Gender_Male", "Smoking Status_Yes", "Location_Suburban", "Location_Urban",
    "Property Type_Condo", "Property Type_House", "Policy Type_Premium",
    "Customer Feedback_Poor", "Customer Feedback_Satisfied", "Customer Feedback_Very Satisfied",
    "Marital Status_Married", "Marital Status_Single", "Occupation_Unknown",
    "Occupation_Professional", "Occupation_Skilled", "Exercise Frequency_None",
    "Exercise Frequency_Regular", "Exercise Frequency_Weekly", "Education Level_High School"
]

# App UI
st.title("🧮 Insurance Premium Prediction App")
st.markdown("Enter your details below to estimate your insurance premium.")

# User inputs
health_score = st.slider("Health Score", 0, 100, 75)
age = st.number_input("Age", min_value=18, max_value=100, value=40)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=680)
vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 3)
insurance_duration = st.slider("Insurance Duration (years)", 0, 20, 5)
annual_income = st.number_input("Annual Income (₦)", min_value=10000, value=500000)
prev_claims = st.number_input("Previous Claims", min_value=0, value=1)
dependents = st.number_input("Number of Dependents", min_value=0, value=2)

# Categorical selections
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking Status", ["Yes", "No"])
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
property_type = st.selectbox("Property Type", ["House", "Condo", "Other"])
policy_type = st.selectbox("Policy Type", ["Premium", "Basic"])
feedback = st.selectbox("Customer Feedback", ["Very Satisfied", "Satisfied", "Poor"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
occupation = st.selectbox("Occupation", ["Professional", "Skilled", "Unknown", "Unemployed"])
exercise = st.selectbox("Exercise Frequency", ["Regular", "Weekly", "None"])
education = st.selectbox("Education Level", ["High School", "Graduate", "Postgraduate"])

# Build full input dictionary with all encoded features
full_input = {
    "Health Score": health_score,
    "Age": age,
    "Credit Score": credit_score,
    "Vehicle Age": vehicle_age,
    "Insurance Duration": insurance_duration,
    "Annual Income": annual_income,
    "Annual Income_log": np.log1p(annual_income),
    "Previous Claims": prev_claims,
    "Previous Claims_log": np.log1p(prev_claims),
    "Number of Dependents": dependents,

    # Encoded categorical values
    "Gender_Male": 1 if gender == "Male" else 0,
    "Smoking Status_Yes": 1 if smoking == "Yes" else 0,
    "Location_Suburban": 1 if location == "Suburban" else 0,
    "Location_Urban": 1 if location == "Urban" else 0,
    "Property Type_Condo": 1 if property_type == "Condo" else 0,
    "Property Type_House": 1 if property_type == "House" else 0,
    "Policy Type_Premium": 1 if policy_type == "Premium" else 0,

    "Customer Feedback_Poor": 1 if feedback == "Poor" else 0,
    "Customer Feedback_Satisfied": 1 if feedback == "Satisfied" else 0,
    "Customer Feedback_Very Satisfied": 1 if feedback == "Very Satisfied" else 0,

    "Marital Status_Married": 1 if marital_status == "Married" else 0,
    "Marital Status_Single": 1 if marital_status == "Single" else 0,

    "Occupation_Unknown": 1 if occupation == "Unknown" else 0,
    "Occupation_Professional": 1 if occupation == "Professional" else 0,
    "Occupation_Skilled": 1 if occupation == "Skilled" else 0,

    "Exercise Frequency_None": 1 if exercise == "None" else 0,
    "Exercise Frequency_Regular": 1 if exercise == "Regular" else 0,
    "Exercise Frequency_Weekly": 1 if exercise == "Weekly" else 0,

    "Education Level_High School": 1 if education == "High School" else 0,
}

# Reorder input features
ordered_input = [full_input.get(feat, 0) for feat in feature_order]

# DataFrame for model input
ordered_df = pd.DataFrame([ordered_input], columns=feature_order)

st.write("📋 INPUT DATAFRAME (ordered features):")
st.write(ordered_df)

# Scale input
scaled_input = scaler.transform(ordered_df)

st.write("🔢 SCALED INPUT (used for prediction):")
st.write(scaled_input)

# Predict using model
raw_pred = model.predict(scaled_input)
st.write("🧠 RAW MODEL OUTPUT (before inverse log):")
st.write(raw_pred)

# Inverse log to get actual premium
final_pred = np.expm1(raw_pred)[0]
st.success(f"💡 Predicted Premium Amount: ₦{final_pred:.2f}")

import streamlit as st
import numpy as np
import joblib
import os

# Load the model and scaler
model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_order.pkl")

# Define realistic defaults for numerical features
realistic_defaults = {
    "Health Score": 75.0,
    "Age": 35.0,
    "Credit Score": 650.0,
    "Vehicle Age": 3.0,
    "Annual Income": 60000.0,
    "Annual Income_log": np.log1p(60000.0),
    "Insurance Duration": 2.0,
    "Number of Dependents": 2.0,
    "Previous Claims": 1.0,
    "Previous Claims_log": np.log1p(1.0),
    "Premium Amount_log": 10.0  # Not user input
}

# Define default values (1 for common categories, 0 otherwise)
category_defaults = {
    "Gender_Male": 1.0,
    "Smoking Status_Yes": 0.0,
    "Location_Suburban": 1.0,
    "Location_Urban": 0.0,
    "Property Type_Condo": 1.0,
    "Property Type_House": 0.0,
    "Policy Type_Premium": 1.0,
    "Policy Type_Comprehensive": 0.0,
    "Customer Feedback_Poor": 0.0,
    "Customer Feedback_Good": 1.0,
    "Marital Status_Single": 1.0,
    "Marital Status_Married": 0.0,
    "Occupation_Unknown": 0.0,
    "Exercise Frequency_Monthly": 1.0,
    "Exercise Frequency_Rarely": 0.0,
    "Exercise Frequency_Weekly": 0.0,
    "Education Level_High School": 0.0,
    "Education Level_Master's": 1.0,
    "Education Level_PhD": 0.0
}

# Streamlit UI
st.set_page_config(page_title="Insurance Premium Estimator", layout="centered")
st.title("🧮 Insurance Premium Estimator")
st.write("Enter customer details to estimate their insurance premium.")

user_input = {}

# Generate input fields based on feature order
for feature in feature_order:
    if feature == "Premium Amount_log":
        continue  # Don't take this as input
    elif feature in realistic_defaults:
        user_input[feature] = st.number_input(feature, value=round(realistic_defaults[feature], 2))
    else:
        user_input[feature] = st.number_input(feature, value=float(category_defaults.get(feature, 0.0)))

# Predict button
if st.button("Estimate Premium"):
    # Prepare input array in correct order
    input_array = np.array([user_input[feat] for feat in feature_order if feat != "Premium Amount_log"]).reshape(1, -1)

    # Scale the input
    scaled_input = scaler.transform(input_array)

    # Predict log premium
    log_prediction = model.predict(scaled_input)[0]

    # Convert back to original premium value
    predicted_premium = np.expm1(log_prediction)

    st.success(f"💰 Estimated Insurance Premium: **${predicted_premium:,.2f}**")

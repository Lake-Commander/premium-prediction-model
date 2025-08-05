import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, feature order
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_order.pkl")

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("💰 Insurance Premium Prediction App")
st.markdown("Enter customer details below to predict their insurance premium amount.")

# --- Input Sections ---
st.header("🧾 Customer Profile")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoker?", ["Yes", "No"])
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    property_type = st.selectbox("Property Type", ["Condo", "House", "Apartment"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    policy_type = st.selectbox("Policy Type", ["Comprehensive", "Premium", "Basic"])

with col2:
    occupation = st.selectbox("Occupation", ["Professional", "Self-Employed", "Unemployed", "Unknown"])
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    exercise = st.selectbox("Exercise Frequency", ["Rarely", "Monthly", "Weekly", "Daily"])
    feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good", "Excellent"])

# Assume numerical inputs were selected during feature selection
st.header("📈 Numerical Features")
aims_log = st.slider("AIMS Score (Log Transformed)", -2.0, 2.0, 0.0, 0.01)

# --- Process Inputs ---
def preprocess_input():
    # Create initial feature dict with all zeroes
    input_dict = {feat: 0 for feat in feature_order}

    # Fill known numerical
    input_dict["aims_log"] = aims_log

    # One-hot encode based on known dummies in feature_order
    mapping = {
        f"Gender_{gender}": 1,
        f"Smoking Status_{smoking}": 1,
        f"Location_{location}": 1,
        f"Property Type_{property_type}": 1,
        f"Marital Status_{marital_status}": 1,
        f"Policy Type_{policy_type}": 1,
        f"Customer Feedback_{feedback}": 1,
        f"Education Level_{education}": 1,
        f"Exercise Frequency_{exercise}": 1,
        f"Occupation_{occupation}": 1,
    }

    for key in mapping:
        if key in input_dict:
            input_dict[key] = mapping[key]

    # Convert to dataframe with correct feature order
    input_df = pd.DataFrame([input_dict])[feature_order]

    # Scale input
    scaled_input = scaler.transform(input_df)

    return scaled_input

# --- Predict ---
if st.button("🔮 Predict Premium Amount"):
    X_input = preprocess_input()
    prediction = model.predict(X_input)[0]
    st.success(f"💡 Predicted Premium Amount: **₦{prediction:,.2f}**")

# --- Optional: Feature Importance ---
if hasattr(model, "feature_importances_"):
    st.header("📊 Feature Importances")
    importance_df = pd.DataFrame({
        "Feature": feature_order,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)
    st.bar_chart(importance_df.set_index("Feature"))

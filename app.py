import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and feature order
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_order.pkl")

# Title
st.title("Insurance Premium Prediction")
st.subheader("Enter Customer Information")

# Numerical fields
numerical_fields = [
    'Health Score', 'Age', 'Credit Score', 'Vehicle Age', 'Annual Income_log',
    'Annual Income', 'Insurance Duration', 'Number of Dependents',
    'Previous Claims', 'Previous Claims_log'
]

# Binary/encoded fields
binary_fields = feature_order.copy()
for field in numerical_fields:
    if field in binary_fields:
        binary_fields.remove(field)

# UI inputs
user_inputs = {}

# Input for numerical fields
for field in numerical_fields:
    user_inputs[field] = st.number_input(field, value=0.0, format="%.2f")

# Input for binary fields (checkbox for 0/1)
for field in binary_fields:
    user_inputs[field] = 1.0 if st.checkbox(field) else 0.0

# Prepare input in correct order
input_data = pd.DataFrame([[user_inputs.get(col, 0.0) for col in feature_order]], columns=feature_order)

# Prediction
if st.button("Predict Premium"):
    try:
        input_scaled = scaler.transform(input_data)
        log_pred = model.predict(input_scaled)[0]
        final_pred = np.expm1(log_pred)
        st.success(f"Estimated Premium: ₦{final_pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

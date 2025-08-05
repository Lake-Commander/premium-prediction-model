import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and dependencies
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_order.pkl")

# App UI
st.title("🚀 Insurance Premium Prediction")
st.write("Enter customer details to estimate their insurance premium.")

# Dynamically generate input fields
user_input = {}
for col in feature_columns:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict Premium"):
    prediction = model.predict(scaled_input)
    st.success(f"💰 Predicted Premium: ₦{prediction[0]:,.2f}")

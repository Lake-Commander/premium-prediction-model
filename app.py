import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load trained components ===
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_order = joblib.load('models/feature_order.pkl')

# === Title ===
st.title("Insurance Premium Prediction")

# === Input form ===
st.header("Enter Customer Information")

# Collect all input features (adjust based on your dataset)
input_data = {}
for feature in feature_order:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Reorder to match training feature order
input_df = input_df.reindex(columns=feature_order, fill_value=0)

# === Predict ===
if st.button("Predict Premium"):
    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict log premium
    log_pred = model.predict(scaled_input)

    # Convert back to original scale
    predicted_premium = np.expm1(log_pred[0])

    st.success(f"Predicted Premium Amount: ₦{predicted_premium:,.2f}")

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and dependencies
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_order.pkl")

# Define realistic options for encoded features
categorical_mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Vehicle_Age': {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2},
    'Vehicle_Damage': {'No': 0, 'Yes': 1},
    'City_Type': {'Urban': 0, 'Rural': 1},
    'Region_Code': {'Region_1': 1, 'Region_2': 2, 'Region_3': 3},  # Update with actual codes if known
}

# Start Streamlit App
st.title("🚀 Insurance Premium Prediction")
st.write("Enter customer details to estimate their insurance premium.")

user_input = {}

# Generate UI
for col in feature_columns:
    if col in categorical_mappings:
        # Show dropdown with readable labels
        label_map = categorical_mappings[col]
        selected = st.selectbox(f"{col}", list(label_map.keys()))
        user_input[col] = label_map[selected]
    else:
        # For numerical input fields
        user_input[col] = st.number_input(f"{col}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Ensure column order matches
input_df = input_df[feature_columns]

# Predict
if st.button("Predict Premium"):
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        st.success(f"💰 Predicted Premium: ₦{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

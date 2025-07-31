# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature names
model = joblib.load('random_forest_model.pkl')

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("💰 Insurance Premium Prediction App")

st.markdown("Fill out the form below to predict the **Premium Amount**.")

# UI for user input
def user_input_features():
    # Example input fields – adjust based on your actual dataset
    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city_type = st.selectbox("City Type", ["Urban", "Semi-Urban", "Rural"])
    is_smoker = st.selectbox("Smoker", ["Yes", "No"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.slider("Number of Children", 0, 5, 1)

    data = {
        "Age": age,
        "Gender": gender,
        "City_Type": city_type,
        "Smoker": is_smoker,
        "BMI": bmi,
        "Children": children
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Combine input with feature columns
def preprocess_input(input_df):
    # Load full feature list from training
    df = pd.DataFrame(columns=feature_names)

    # Merge with user input
    df = pd.concat([df, input_df], ignore_index=True)

    # Handle encoding
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes

    # Fill any missing columns with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[feature_names]
    return df

if st.button("Predict Premium"):
    input_processed = preprocess_input(input_df)
    prediction = model.predict(input_processed)[0]
    st.success(f"🏷️ Estimated Premium Amount: ₦{prediction:,.2f}")

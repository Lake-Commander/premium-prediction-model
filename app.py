import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("random_forest_model.pkl")

# If you used encoders or scalers, load them here
# encoder = joblib.load("encoder.pkl")  # optional
# scaler = joblib.load("scaler.pkl")    # optional

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("💰 Insurance Premium Prediction App")
st.markdown("Enter the details below to estimate your insurance premium.")

# User input form
age = st.slider("Age", 18, 100, 35)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 60.0, 28.5)
children = st.slider("Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
city_type = st.selectbox("City Type", ["Urban", "Rural"])  # only if relevant

# Create raw input DataFrame
input_dict = {
    "age": age,
    "sex": gender.lower(),
    "bmi": bmi,
    "children": children,
    "smoker": smoker.lower(),
    "region": region.lower(),
    "city_type": city_type.lower()  # include only if used in training
}
input_df = pd.DataFrame([input_dict])

# Manual preprocessing (same as training)
def preprocess_input(df):
    df = df.copy()

    # One-hot encode categorical variables
    df = pd.get_dummies(df)

    # Ensure all training features are present
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[model.feature_names_in_]
    return df

if st.button("Predict Premium"):
    try:
        # Preprocess
        X_input = preprocess_input(input_df)

        # Predict log-premium
        log_pred = model.predict(X_input)[0]

        # Inverse log transformation
        premium = np.exp(log_pred)

        st.success(f"💸 Estimated Premium Amount: ₦{premium:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

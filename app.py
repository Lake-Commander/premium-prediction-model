import streamlit as st
import numpy as np
import pandas as pd

# Placeholder: Simulate a trained model
def dummy_model_predict(X):
    return 25000 + (X[:, 0] * 500)  # Just a placeholder logic

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

st.title("💸 Insurance Premium Prediction")
st.markdown("Fill in the details to estimate your annual premium.")

with st.sidebar:
    st.header("📝 Client Info")

    age = st.slider("Age", 18, 100, 30)
    health_score = st.slider("Health Score", 0, 100, 70)
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    vehicle_age = st.number_input("Vehicle Age (Years)", 0, 20, 5)
    income = st.number_input("Annual Income (₦)", 100_000, 50_000_000, 1_000_000, step=100000)
    insurance_duration = st.slider("Insurance Duration (Years)", 1, 30, 5)
    num_dependents = st.slider("Number of Dependents", 0, 10, 2)
    prev_claims = st.number_input("Previous Claims", 0, 100, 0)
    exercise_freq = st.slider("Exercise Frequency (Days/Week)", 0, 7, 3)

    gender = st.radio("Gender", ["Male", "Female"])
    smoker = st.radio("Smoking Status", ["Yes", "No"])
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    property_type = st.selectbox("Property Type", ["House", "Condo", "Apartment"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Premium", "Plus"])
    feedback = st.selectbox("Customer Feedback", ["Good", "Average", "Poor"])
    marital_status = st.radio("Marital Status", ["Single", "Married"])
    occupation_unknown = st.checkbox("Occupation Unknown")

# Simulate transformation (manually create one-hot encodings for this example)
def process_input():
    data = {
        "Age": age,
        "Health Score": health_score,
        "Credit Score": credit_score,
        "Vehicle Age": vehicle_age,
        "Annual Income_log": np.log1p(income),
        "Insurance Duration": insurance_duration,
        "Number of Dependents": num_dependents,
        "Previous Claims_log": np.log1p(prev_claims),
        "Exercise Frequency": exercise_freq,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Smoking Status_Yes": 1 if smoker == "Yes" else 0,
        "Location_Urban": 1 if location == "Urban" else 0,
        "Location_Suburban": 1 if location == "Suburban" else 0,
        "Property Type_House": 1 if property_type == "House" else 0,
        "Property Type_Condo": 1 if property_type == "Condo" else 0,
        "Policy Type_Premium": 1 if policy_type == "Premium" else 0,
        "Policy Type_Plus": 1 if policy_type == "Plus" else 0,
        "Customer Feedback_Poor": 1 if feedback == "Poor" else 0,
        "Customer Feedback_Average": 1 if feedback == "Average" else 0,
        "Marital Status_Single": 1 if marital_status == "Single" else 0,
        "Marital Status_Married": 1 if marital_status == "Married" else 0,
        "Occupation_Unknown": 1 if occupation_unknown else 0,
    }

    X = pd.DataFrame([data])
    return X

if st.button("🔮 Predict Premium"):
    X = process_input()
    X_np = X.values
    prediction = dummy_model_predict(X_np)

    st.success(f"Estimated Premium: ₦{prediction[0]:,.2f}")
    st.caption("Note: This is a simulated output. Actual prediction will be more precise after model integration.")


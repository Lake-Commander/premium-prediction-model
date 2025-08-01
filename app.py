import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return joblib.load(f)

model = load_model()

# List of all expected features (used during training)
FEATURES = [
    'Premium Amount_log', 'Health Score', 'Age', 'Credit Score', 'Vehicle Age',
    'Annual Income_log', 'Annual Income', 'Insurance Duration',
    'Number of Dependents', 'Previous Claims', 'Previous Claims_log',
    'Gender_Male', 'Smoking Status_Yes', 'Location_Suburban',
    'Property Type_Condo', 'Location_Urban', 'Policy Type_Premium',
    'Customer Feedback_Poor', 'Marital Status_Single', 'Property Type_House',
    'Occupation_Unknown', 'Marital Status_Married',
    'Exercise Frequency_Monthly', 'Exercise Frequency_Rarely',
    'Education Level_PhD', 'Customer Feedback_Good',
    'Policy Type_Comprehensive', "Education Level_Master's",
    'Exercise Frequency_Weekly', 'Education Level_High School'
]

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("💸 Insurance Premium Prediction App")
st.markdown("Enter your details below to predict the premium amount (log-scaled).")

# === User Input ===
with st.form("prediction_form"):
    st.subheader("Basic Information")
    age = st.number_input("Age", 18, 100, step=1)
    health_score = st.slider("Health Score (0-100)", 0, 100, value=70)
    credit_score = st.slider("Credit Score (300-850)", 300, 850, value=600)
    vehicle_age = st.number_input("Vehicle Age (years)", 0, 50, step=1)
    insurance_duration = st.number_input("Insurance Duration (years)", 0.0, 50.0, step=0.5)
    num_dependents = st.number_input("Number of Dependents", 0, 10, step=1)
    prev_claims = st.number_input("Number of Previous Claims", 0, 20, step=1)
    annual_income = st.number_input("Annual Income ($)", 1000, 1_000_000, step=1000)

    st.subheader("Lifestyle & Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoker = st.selectbox("Smoking Status", ["Yes", "No"])
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Unknown", "Employed", "Unemployed", "Student"])
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"])
    property_type = st.selectbox("Property Type", ["Apartment", "House", "Condo"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    exercise = st.selectbox("Exercise Frequency", ["Rarely", "Monthly", "Weekly", "Daily"])

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    # === Construct feature vector ===
    data = {
        'Premium Amount_log': 0,  # Placeholder, will not affect prediction
        'Health Score': health_score,
        'Age': age,
        'Credit Score': credit_score,
        'Vehicle Age': vehicle_age,
        'Annual Income_log': np.log1p(annual_income),
        'Annual Income': annual_income,
        'Insurance Duration': insurance_duration,
        'Number of Dependents': num_dependents,
        'Previous Claims': prev_claims,
        'Previous Claims_log': np.log1p(prev_claims),
        'Gender_Male': int(gender == "Male"),
        'Smoking Status_Yes': int(smoker == "Yes"),
        'Location_Suburban': int(location == "Suburban"),
        'Location_Urban': int(location == "Urban"),
        'Property Type_Condo': int(property_type == "Condo"),
        'Property Type_House': int(property_type == "House"),
        'Policy Type_Premium': int(policy_type == "Premium"),
        'Policy Type_Comprehensive': int(policy_type == "Comprehensive"),
        'Customer Feedback_Poor': int(feedback == "Poor"),
        'Customer Feedback_Good': int(feedback == "Good"),
        'Marital Status_Single': int(marital_status == "Single"),
        'Marital Status_Married': int(marital_status == "Married"),
        'Occupation_Unknown': int(occupation == "Unknown"),
        'Exercise Frequency_Monthly': int(exercise == "Monthly"),
        'Exercise Frequency_Rarely': int(exercise == "Rarely"),
        'Exercise Frequency_Weekly': int(exercise == "Weekly"),
        'Education Level_High School': int(education == "High School"),
        "Education Level_Master's": int(education == "Master's"),
        'Education Level_PhD': int(education == "PhD"),
    }

    input_df = pd.DataFrame([data])

    # Ensure all 30 features are present in correct order
    for col in FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[FEATURES]

    # Make prediction
    prediction_log = model.predict(input_df)[0]
    predicted_amount = np.expm1(prediction_log)

    st.success(f"💰 Estimated Premium Amount: **${predicted_amount:,.2f}**")

    st.caption("Note: This prediction is based on a machine learning model and may not reflect actual insurance pricing.")

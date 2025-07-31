import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math

# Load model
model = joblib.load("random_forest_model.pkl")

# Define exact columns expected by the model (must match training)
model_features = [
    'Health Score', 'Age', 'Credit Score', 'Vehicle Age',
    'Annual Income_log', 'Annual Income', 'Insurance Duration',
    'Number of Dependents', 'Previous Claims', 'Previous Claims_log',
    'Gender_Male', 'Smoking Status_Yes', 'Location_Suburban', 'Property Type_Condo',
    'Location_Urban', 'Policy Type_Premium', 'Customer Feedback_Poor',
    'Marital Status_Single', 'Property Type_House', 'Occupation_Unknown',
    'Marital Status_Married', 'Exercise Frequency_Monthly', 'Exercise Frequency_Rarely',
    'Education Level_PhD', 'Customer Feedback_Good', 'Policy Type_Comprehensive',
    "Education Level_Master's", 'Exercise Frequency_Weekly',
    'Education Level_High School'
]

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("💼 Insurance Premium Predictor")

st.markdown("Fill in client details to estimate the insurance premium.")

with st.form("predict_form"):
    age = st.slider("Age", 18, 100, 35)
    health_score = st.slider("Health Score", 0.0, 100.0, 50.0)
    credit_score = st.slider("Credit Score", 300, 850, 600)
    vehicle_age = st.slider("Vehicle Age (years)", 0, 25, 5)
    annual_income = st.number_input("Annual Income (₦)", 10000, 10000000, 500000)
    insurance_duration = st.slider("Insurance Duration (years)", 1, 30, 5)
    dependents = st.slider("Number of Dependents", 0, 10, 2)
    previous_claims = st.slider("Number of Previous Claims", 0, 20, 1)

    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking Status", ["Yes", "No"])
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    property_type = st.selectbox("Property Type", ["Condo", "House", "Apartment"])
    policy_type = st.selectbox("Policy Type", ["Comprehensive", "Premium", "Basic"])
    feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Unknown", "Employed", "Self-Employed"])
    exercise = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    education = st.selectbox("Education Level", ["PhD", "Master's", "High School", "Bachelor's"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Derived columns
    annual_income_log = np.log1p(annual_income)
    previous_claims_log = np.log1p(previous_claims)

    # One-hot encoded fields (initialize all to 0, only activate selected)
    input_dict = dict.fromkeys(model_features, 0)

    input_dict.update({
        "Health Score": health_score,
        "Age": age,
        "Credit Score": credit_score,
        "Vehicle Age": vehicle_age,
        "Annual Income_log": annual_income_log,
        "Annual Income": annual_income,
        "Insurance Duration": insurance_duration,
        "Number of Dependents": dependents,
        "Previous Claims": previous_claims,
        "Previous Claims_log": previous_claims_log,
    })

    # Manually encode one-hot features
    if gender == "Male":
        input_dict["Gender_Male"] = 1
    if smoking == "Yes":
        input_dict["Smoking Status_Yes"] = 1
    if location == "Urban":
        input_dict["Location_Urban"] = 1
    elif location == "Suburban":
        input_dict["Location_Suburban"] = 1
    if property_type == "Condo":
        input_dict["Property Type_Condo"] = 1
    elif property_type == "House":
        input_dict["Property Type_House"] = 1
    if policy_type == "Premium":
        input_dict["Policy Type_Premium"] = 1
    elif policy_type == "Comprehensive":
        input_dict["Policy Type_Comprehensive"] = 1
    if feedback == "Poor":
        input_dict["Customer Feedback_Poor"] = 1
    elif feedback == "Good":
        input_dict["Customer Feedback_Good"] = 1
    if marital_status == "Single":
        input_dict["Marital Status_Single"] = 1
    elif marital_status == "Married":
        input_dict["Marital Status_Married"] = 1
    if occupation == "Unknown":
        input_dict["Occupation_Unknown"] = 1
    if exercise == "Monthly":
        input_dict["Exercise Frequency_Monthly"] = 1
    elif exercise == "Rarely":
        input_dict["Exercise Frequency_Rarely"] = 1
    elif exercise == "Weekly":
        input_dict["Exercise Frequency_Weekly"] = 1
    if education == "PhD":
        input_dict["Education Level_PhD"] = 1
    elif education == "Master's":
        input_dict["Education Level_Master's"] = 1
    elif education == "High School":
        input_dict["Education Level_High School"] = 1

    # Final DataFrame for prediction
    input_df = pd.DataFrame([input_dict])[model_features]

    # Prediction
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💸 Estimated Premium: **₦{prediction:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

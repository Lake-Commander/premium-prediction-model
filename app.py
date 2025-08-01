import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define the exact feature order used during training
feature_order = [
    'Health Score', 'Age', 'Credit Score', 'Vehicle Age', 'Annual Income_log',
    'Insurance Duration', 'Number of Dependents', 'Previous Claims_log',
    'Gender_Male', 'Smoking Status_Yes', 'Location_Suburban', 'Location_Urban',
    'Property Type_Condo', 'Property Type_House', 'Policy Type_Premium',
    'Customer Feedback_Poor', 'Marital Status_Married', 'Marital Status_Single',
    'Occupation_Unknown', 'Exercise Frequency', 'Education Level_High School',
    'Education Level_Postgraduate', 'Education Level_Undergraduate',
    'Vehicle Type_SUV', 'Vehicle Type_Sedan', 'Vehicle Type_Truck',
    'Region_North', 'Region_South', 'Region_East', 'Region_West'
]

# Title
st.set_page_config(page_title="Insurance Premium Estimator")
st.title("💼 Insurance Premium Prediction App")

# Sidebar
st.sidebar.header("Client Information")

# Input form
def user_input_features():
    health_score = st.sidebar.slider("Health Score", 0, 100, 70)
    age = st.sidebar.slider("Age", 18, 100, 35)
    credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
    vehicle_age = st.sidebar.number_input("Vehicle Age (years)", 0, 30, 5)
    annual_income = st.sidebar.number_input("Annual Income ($)", 1000, 1000000, 50000)
    insurance_duration = st.sidebar.slider("Insurance Duration (years)", 0, 30, 3)
    num_dependents = st.sidebar.slider("Number of Dependents", 0, 10, 1)
    previous_claims = st.sidebar.number_input("Previous Claims", 0, 100, 0)
    exercise_freq = st.sidebar.slider("Exercise Frequency (days/week)", 0, 7, 3)

    # Categorical selections
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    smoker = st.sidebar.radio("Smoking Status", ["Yes", "No"])
    location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])
    property_type = st.sidebar.selectbox("Property Type", ["House", "Condo", "Apartment"])
    policy_type = st.sidebar.selectbox("Policy Type", ["Basic", "Standard", "Premium"])
    feedback = st.sidebar.selectbox("Customer Feedback", ["Excellent", "Good", "Average", "Poor"])
    marital_status = st.sidebar.radio("Marital Status", ["Single", "Married"])
    occupation = st.sidebar.selectbox("Occupation", ["Professional", "Blue Collar", "White Collar", "Unknown"])
    education = st.sidebar.selectbox("Education Level", ["High School", "Undergraduate", "Postgraduate"])
    vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Sedan", "SUV", "Truck"])
    region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])

    # Manual encoding
    data = {
        'Health Score': health_score,
        'Age': age,
        'Credit Score': credit_score,
        'Vehicle Age': vehicle_age,
        'Annual Income_log': np.log1p(annual_income),
        'Insurance Duration': insurance_duration,
        'Number of Dependents': num_dependents,
        'Previous Claims_log': np.log1p(previous_claims),
        'Exercise Frequency': exercise_freq,

        'Gender_Male': 1 if gender == "Male" else 0,
        'Smoking Status_Yes': 1 if smoker == "Yes" else 0,
        'Location_Suburban': 1 if location == "Suburban" else 0,
        'Location_Urban': 1 if location == "Urban" else 0,

        'Property Type_Condo': 1 if property_type == "Condo" else 0,
        'Property Type_House': 1 if property_type == "House" else 0,

        'Policy Type_Premium': 1 if policy_type == "Premium" else 0,
        'Customer Feedback_Poor': 1 if feedback == "Poor" else 0,

        'Marital Status_Single': 1 if marital_status == "Single" else 0,
        'Marital Status_Married': 1 if marital_status == "Married" else 0,

        'Occupation_Unknown': 1 if occupation == "Unknown" else 0,

        'Education Level_High School': 1 if education == "High School" else 0,
        'Education Level_Postgraduate': 1 if education == "Postgraduate" else 0,
        'Education Level_Undergraduate': 1 if education == "Undergraduate" else 0,

        'Vehicle Type_SUV': 1 if vehicle_type == "SUV" else 0,
        'Vehicle Type_Sedan': 1 if vehicle_type == "Sedan" else 0,
        'Vehicle Type_Truck': 1 if vehicle_type == "Truck" else 0,

        'Region_North': 1 if region == "North" else 0,
        'Region_South': 1 if region == "South" else 0,
        'Region_East': 1 if region == "East" else 0,
        'Region_West': 1 if region == "West" else 0
    }

    return pd.DataFrame([data])

# Predict button
if st.button("Estimate Premium"):
    input_df = user_input_features()

    # Ensure correct column order
    input_df = input_df.reindex(columns=feature_order)

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict and inverse log transform
    log_prediction = model.predict(scaled_input)
    premium = np.expm1(log_prediction[0])

    st.success(f"💰 Estimated Premium Amount: ${premium:,.2f}")

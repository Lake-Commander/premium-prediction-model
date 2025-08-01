import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('random_forest_model.pkl')

# Title
st.title("💼 Insurance Premium Predictor")

# Sidebar instructions
st.sidebar.header("📋 Customer Information")

# Collect inputs
premium_log = st.sidebar.number_input("Premium Amount (log scale)", value=8.3)
health_score = st.sidebar.slider("Health Score", 0, 100, 75)
age = st.sidebar.slider("Age", 18, 100, 35)
credit_score = st.sidebar.number_input("Credit Score", value=680)
vehicle_age = st.sidebar.slider("Vehicle Age", 0, 20, 5)
income_log = st.sidebar.number_input("Annual Income (log scale)", value=10.6)
insurance_duration = st.sidebar.slider("Insurance Duration (years)", 0, 50, 4)
dependents = st.sidebar.slider("Number of Dependents", 0, 10, 2)
claims_log = st.sidebar.number_input("Previous Claims (log scale)", value=1.6)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
smoking = st.sidebar.selectbox("Smoking Status", ["Yes", "No"])
location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])
property_type = st.sidebar.selectbox("Property Type", ["Condo", "House", "Other"])
policy_type = st.sidebar.selectbox("Policy Type", ["Premium", "Basic"])
feedback = st.sidebar.selectbox("Customer Feedback", ["Poor", "Good"])
marital = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
occupation = st.sidebar.selectbox("Occupation", ["Known", "Unknown"])
exercise = st.sidebar.slider("Exercise Frequency (days/week)", 0, 7, 3)

# Encode categorical features
gender_male = 1 if gender == "Male" else 0
smoking_yes = 1 if smoking == "Yes" else 0
location_suburban = 1 if location == "Suburban" else 0
location_urban = 1 if location == "Urban" else 0
property_condo = 1 if property_type == "Condo" else 0
property_house = 1 if property_type == "House" else 0
policy_premium = 1 if policy_type == "Premium" else 0
feedback_poor = 1 if feedback == "Poor" else 0
marital_single = 1 if marital == "Single" else 0
marital_married = 1 if marital == "Married" else 0
occupation_unknown = 1 if occupation == "Unknown" else 0

# Combine all inputs
X = np.array([
    premium_log, health_score, age, credit_score, vehicle_age,
    income_log, insurance_duration, dependents, claims_log,
    gender_male, smoking_yes, location_suburban, property_condo,
    location_urban, policy_premium, feedback_poor, marital_single,
    property_house, occupation_unknown, marital_married, exercise
]).reshape(1, -1)

# Predict
if st.button("🚀 Predict Premium"):
    pred_log = model.predict(X)[0]
    pred_actual = np.exp(pred_log)
    st.success(f"💰 Predicted Premium: ${pred_actual:,.2f}")

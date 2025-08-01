import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# List of features in the correct order
features_order = [
    'Health Score', 'Age', 'Credit Score', 'Vehicle Age', 'Annual Income_log',
    'Insurance Duration', 'Number of Dependents', 'Previous Claims_log',
    'Gender_Male', 'Smoking Status_Yes', 'Location_Suburban', 'Property Type_Condo',
    'Location_Urban', 'Policy Type_Premium', 'Customer Feedback_Poor',
    'Marital Status_Single', 'Property Type_House', 'Occupation_Unknown',
    'Marital Status_Married', 'Exercise Frequency'
]

# Set up the UI
st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("💸 Insurance Premium Prediction")
st.write("Enter customer details below to predict the expected premium amount.")

with st.form("input_form"):
    st.subheader("📋 Basic Information")
    age = st.slider("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking Status", ["Yes", "No"])
    health_score = st.slider("Health Score", 1, 10, 5)
    exercise_freq = st.slider("Exercise Frequency (days/week)", 0, 7, 3)

    st.subheader("🏠 Lifestyle & Property")
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    num_dependents = st.slider("Number of Dependents", 0, 10, 2)
    property_type = st.selectbox("Property Type", ["House", "Condo", "Apartment"])
    location = st.selectbox("Location Type", ["Urban", "Suburban", "Rural"])

    st.subheader("💼 Financial & Insurance")
    income = st.number_input("Annual Income (₦)", min_value=10000.0, value=300000.0)
    credit_score = st.slider("Credit Score", 300, 850, 600)
    vehicle_age = st.slider("Vehicle Age (years)", 0, 30, 5)
    prev_claims = st.slider("Number of Previous Claims", 0, 10, 1)
    insurance_duration = st.slider("Insurance Duration (years)", 1, 20, 5)
    policy_type = st.selectbox("Policy Type", ["Basic", "Premium"])
    feedback = st.selectbox("Customer Feedback", ["Good", "Poor"])
    occupation = st.selectbox("Occupation", ["Employed", "Unemployed", "Unknown"])

    submit = st.form_submit_button("Predict")

# Handle prediction
if submit:
    # Encode categorical inputs
    data = {
        'Health Score': health_score,
        'Age': age,
        'Credit Score': credit_score,
        'Vehicle Age': vehicle_age,
        'Annual Income_log': np.log1p(income),
        'Insurance Duration': insurance_duration,
        'Number of Dependents': num_dependents,
        'Previous Claims_log': np.log1p(prev_claims),
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Smoking Status_Yes': 1 if smoking == 'Yes' else 0,
        'Location_Suburban': 1 if location == 'Suburban' else 0,
        'Location_Urban': 1 if location == 'Urban' else 0,
        'Property Type_Condo': 1 if property_type == 'Condo' else 0,
        'Property Type_House': 1 if property_type == 'House' else 0,
        'Policy Type_Premium': 1 if policy_type == 'Premium' else 0,
        'Customer Feedback_Poor': 1 if feedback == 'Poor' else 0,
        'Marital Status_Single': 1 if marital_status == 'Single' else 0,
        'Marital Status_Married': 1 if marital_status == 'Married' else 0,
        'Occupation_Unknown': 1 if occupation == 'Unknown' else 0,
        'Exercise Frequency': exercise_freq,
    }

    # Ensure order and convert to array
    X = np.array([data[feature] for feature in features_order]).reshape(1, -1)

    # Scale inputs
    X_scaled = scaler.transform(X)

    # Predict and clip
    prediction = model.predict(X_scaled)[0]
    prediction = max(prediction, 0)  # Clip negatives

    # Display result
    st.success(f"💡 Predicted Premium Amount: ₦{prediction:,.2f}")

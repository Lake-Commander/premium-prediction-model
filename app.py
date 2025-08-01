import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="💸 Insurance Premium Predictor", layout="centered")

st.title("💸 Insurance Premium Prediction App")
st.markdown("Enter values for each feature to predict the insurance premium.")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        health_score = st.number_input("Health Score", min_value=0.0)
        age = st.number_input("Age", min_value=0.0)
        credit_score = st.number_input("Credit Score", min_value=0.0)
        vehicle_age = st.number_input("Vehicle Age", min_value=0.0)
        annual_income_log = st.number_input("Annual Income (log)", min_value=0.0)
        annual_income = st.number_input("Annual Income", min_value=0.0)
        insurance_duration = st.number_input("Insurance Duration", min_value=0.0)
        num_dependents = st.number_input("Number of Dependents", min_value=0.0)
        previous_claims = st.number_input("Previous Claims", min_value=0.0)
        previous_claims_log = st.number_input("Previous Claims (log)", min_value=0.0)
        gender_male = st.selectbox("Gender", ["Female", "Male"]) == "Male"
        smoking_status_yes = st.selectbox("Smoker?", ["No", "Yes"]) == "Yes"
        location_suburban = st.selectbox("Location", ["Rural", "Suburban", "Urban"]) == "Suburban"
        location_urban = st.selectbox("Urban?", ["No", "Yes"]) == "Yes"

    with col2:
        property_type_condo = st.selectbox("Property Type (Condo)?", ["No", "Yes"]) == "Yes"
        property_type_house = st.selectbox("Property Type (House)?", ["No", "Yes"]) == "Yes"
        policy_type_premium = st.selectbox("Policy Type (Premium)?", ["No", "Yes"]) == "Yes"
        policy_type_comprehensive = st.selectbox("Policy Type (Comprehensive)?", ["No", "Yes"]) == "Yes"
        feedback_poor = st.selectbox("Customer Feedback (Poor)?", ["No", "Yes"]) == "Yes"
        feedback_good = st.selectbox("Customer Feedback (Good)?", ["No", "Yes"]) == "Yes"
        marital_status_single = st.selectbox("Marital Status (Single)?", ["No", "Yes"]) == "Yes"
        marital_status_married = st.selectbox("Marital Status (Married)?", ["No", "Yes"]) == "Yes"
        occupation_unknown = st.selectbox("Occupation Unknown?", ["No", "Yes"]) == "Yes"
        exercise_monthly = st.selectbox("Exercise Frequency: Monthly?", ["No", "Yes"]) == "Yes"
        exercise_rarely = st.selectbox("Exercise Frequency: Rarely?", ["No", "Yes"]) == "Yes"
        exercise_weekly = st.selectbox("Exercise Frequency: Weekly?", ["No", "Yes"]) == "Yes"
        edu_phd = st.selectbox("Education: PhD?", ["No", "Yes"]) == "Yes"
        edu_masters = st.selectbox("Education: Master's?", ["No", "Yes"]) == "Yes"
        edu_hs = st.selectbox("Education: High School?", ["No", "Yes"]) == "Yes"

    submitted = st.form_submit_button("Predict Premium")

    if submitted:
        features = [
            health_score,
            age,
            credit_score,
            vehicle_age,
            annual_income_log,
            annual_income,
            insurance_duration,
            num_dependents,
            previous_claims,
            previous_claims_log,
            float(gender_male),
            float(smoking_status_yes),
            float(location_suburban),
            float(property_type_condo),
            float(location_urban),
            float(policy_type_premium),
            float(feedback_poor),
            float(marital_status_single),
            float(property_type_house),
            float(occupation_unknown),
            float(marital_status_married),
            float(exercise_monthly),
            float(exercise_rarely),
            float(edu_phd),
            float(feedback_good),
            float(policy_type_comprehensive),
            float(edu_masters),
            float(exercise_weekly),
            float(edu_hs),
        ]

        # Predict log premium
        predicted_log = model.predict([features])[0]

        # Convert back to original scale
        predicted_premium = np.expm1(predicted_log)

        st.success(f"💰 Predicted Insurance Premium: ${predicted_premium:,.2f}")

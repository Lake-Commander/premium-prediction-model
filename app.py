import streamlit as st
import numpy as np
import joblib

# === Load model and scaler ===
@st.cache_resource
def load_model():
    model = joblib.load('models/random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# === App Layout ===
st.set_page_config(page_title="Insurance Premium Predictor", layout="wide")
st.title("💸 Insurance Premium Prediction App")
st.markdown("Enter customer details to estimate their insurance premium.")

# === Input fields ===
with st.form("prediction_form"):
    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        health_score = st.slider("Health Score", 0, 100, 50)
        age = st.slider("Age", 18, 100, 35)
        credit_score = st.slider("Credit Score", 300, 900, 600)
        vehicle_age = st.number_input("Vehicle Age (years)", 0.0, 30.0, 5.0)
        annual_income = st.number_input("Annual Income", 10000, 1000000, 50000)
        insurance_duration = st.slider("Insurance Duration (years)", 0, 30, 5)
        num_dependents = st.slider("Number of Dependents", 0, 10, 2)
        previous_claims = st.slider("Number of Previous Claims", 0, 10, 0)

    with col2:
        gender_male = st.selectbox("Gender", ["Male", "Female"]) == "Male"
        smoking_yes = st.selectbox("Smoking Status", ["No", "Yes"]) == "Yes"
        location_suburban = st.selectbox("Location", ["Urban", "Suburban", "Rural"]) == "Suburban"
        location_urban = st.selectbox("Urban Area?", ["No", "Yes"]) == "Yes"
        property_type_condo = st.selectbox("Property Type", ["House", "Condo", "Apartment"]) == "Condo"
        property_type_house = st.selectbox("Is House?", ["No", "Yes"]) == "Yes"
        marital_status_single = st.selectbox("Marital Status", ["Married", "Single"]) == "Single"
        marital_status_married = st.selectbox("Is Married?", ["No", "Yes"]) == "Yes"

    with col3:
        policy_type_premium = st.selectbox("Policy Type", ["Standard", "Premium"]) == "Premium"
        feedback_poor = st.selectbox("Customer Feedback", ["Good", "Average", "Poor"]) == "Poor"
        occupation_unknown = st.selectbox("Occupation Unknown?", ["No", "Yes"]) == "No"
        exercise_freq = st.slider("Exercise Frequency (days/week)", 0, 7, 3)

    submitted = st.form_submit_button("Predict Premium")

# === Predict ===
if submitted:
    # === Derived features ===
    annual_income_log = np.log1p(annual_income)
    previous_claims_log = np.log1p(previous_claims)

    input_data = np.array([
        health_score, age, credit_score, vehicle_age, 
        annual_income_log, annual_income, insurance_duration, 
        num_dependents, previous_claims, previous_claims_log,
        int(gender_male), int(smoking_yes), int(location_suburban),
        int(property_type_condo), int(location_urban), int(policy_type_premium),
        int(feedback_poor), int(marital_status_single), int(property_type_house),
        int(occupation_unknown == "Yes"), int(marital_status_married),
        exercise_freq
    ])

    # Pad zeros if original model had more features (placeholder)
    if input_data.shape[0] < scaler.n_features_in_:
        padding = np.zeros(scaler.n_features_in_ - input_data.shape[0])
        input_data = np.concatenate([input_data, padding])

    input_scaled = scaler.transform([input_data])
    log_prediction = model.predict(input_scaled)[0]
    premium = np.expm1(log_prediction)

    st.success(f"💰 **Estimated Premium Amount:** ${premium:,.2f}")

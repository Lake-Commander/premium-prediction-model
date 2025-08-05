import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model, scaler, and features ===
st.markdown("🔄 Loading model, scaler, and top features...")
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
top_features = joblib.load("models/feature_order.pkl")

st.success("✅ Loaded model")
st.success("✅ Loaded scaler")
st.success(f"✅ Loaded top features: {len(top_features)} features")

# === Create Input Form ===
st.title("💼 Insurance Premium Prediction")
st.write("Fill in the details below to predict the expected premium amount.")

with st.form("input_form"):
    health_score = st.slider("Health Score", 0, 100, 75)
    age = st.slider("Age", 18, 100, 40)
    credit_score = st.slider("Credit Score", 300, 850, 680)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0.0, max_value=30.0, value=3.0)
    annual_income = st.number_input("Annual Income (₦)", min_value=10000.0, value=5_000_000.0)
    income_log = np.log1p(annual_income)

    insurance_duration = st.slider("Insurance Duration (years)", 0, 50, 3)
    dependents = st.slider("Number of Dependents", 0, 10, 1)
    prev_claims = st.slider("Previous Claims", 0, 5, 1)
    prev_claims_log = np.log1p(prev_claims)

    # One-hot encoded categorical fields
    gender_male = st.selectbox("Gender", ["Male", "Female"]) == "Male"
    smoker_yes = st.selectbox("Smoking Status", ["Yes", "No"]) == "Yes"

    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    loc_suburban = location == "Suburban"
    loc_urban = location == "Urban"

    property_type = st.selectbox("Property Type", ["House", "Condo", "Apartment"])
    prop_condo = property_type == "Condo"
    prop_house = property_type == "House"

    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    policy_premium = policy_type == "Premium"
    policy_comprehensive = policy_type == "Comprehensive"

    feedback = st.selectbox("Customer Feedback", ["Good", "Average", "Poor"])
    feedback_poor = feedback == "Poor"
    feedback_good = feedback == "Good"

    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    ms_single = marital_status == "Single"
    ms_married = marital_status == "Married"

    occupation_unknown = st.checkbox("Occupation Unknown")

    exercise_freq = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    ex_weekly = exercise_freq == "Weekly"
    ex_monthly = exercise_freq == "Monthly"
    ex_rarely = exercise_freq == "Rarely"

    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    edu_hs = education == "High School"
    edu_masters = education == "Master's"
    edu_phd = education == "PhD"

    submitted = st.form_submit_button("Predict")

# === Prepare Input and Predict ===
if submitted:
    input_data = {
        'Health Score': health_score,
        'Age': age,
        'Credit Score': credit_score,
        'Vehicle Age': vehicle_age,
        'Annual Income_log': income_log,
        'Annual Income': annual_income,
        'Insurance Duration': insurance_duration,
        'Number of Dependents': dependents,
        'Previous Claims': prev_claims,
        'Previous Claims_log': prev_claims_log,
        'Gender_Male': int(gender_male),
        'Smoking Status_Yes': int(smoker_yes),
        'Location_Suburban': int(loc_suburban),
        'Property Type_Condo': int(prop_condo),
        'Location_Urban': int(loc_urban),
        'Policy Type_Premium': int(policy_premium),
        'Customer Feedback_Poor': int(feedback_poor),
        'Marital Status_Single': int(ms_single),
        'Property Type_House': int(prop_house),
        'Occupation_Unknown': int(occupation_unknown),
        'Marital Status_Married': int(ms_married),
        'Exercise Frequency_Monthly': int(ex_monthly),
        'Exercise Frequency_Rarely': int(ex_rarely),
        'Education Level_PhD': int(edu_phd),
        'Customer Feedback_Good': int(feedback_good),
        'Policy Type_Comprehensive': int(policy_comprehensive),
        "Education Level_Master's": int(edu_masters),
        'Exercise Frequency_Weekly': int(ex_weekly),
        'Education Level_High School': int(edu_hs),
        # DO NOT include 'Premium Amount' — that’s the target!
    }

    # Keep only expected features (no target)
    cleaned_input = {feat: input_data.get(feat, 0) for feat in top_features}
    ordered_df = pd.DataFrame([cleaned_input], columns=top_features)

    # === Feature sanity check ===
    if hasattr(scaler, 'feature_names_in_'):
        missing = set(scaler.feature_names_in_) - set(ordered_df.columns)
        extra = set(ordered_df.columns) - set(scaler.feature_names_in_)
        if missing or extra:
            st.error(f"❌ Feature mismatch. Missing: {missing}, Extra: {extra}")
    try:
        scaled_input = scaler.transform(ordered_df)
        st.write("🔢 SCALED INPUT (used for prediction):")
        st.dataframe(pd.DataFrame(scaled_input, columns=top_features))

        raw_output = model.predict(scaled_input)
        st.write("🧠 RAW MODEL OUTPUT:")
        st.write(raw_output)

        prediction = max(0, raw_output[0])
        st.success(f"💡 Predicted Premium Amount: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

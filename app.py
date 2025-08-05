import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ======================================
# 📊 EDA DASHBOARD SECTION
# ======================================
st.set_page_config(page_title="Insurance Premium Predictor", layout="wide")

st.title("📊 Insurance Premium EDA Dashboard")

def show_images_from_folder(folder, title=None):
    if title:
        st.subheader(title)
    if not os.path.exists(folder):
        st.warning(f"Folder not found: {folder}")
        return
    for file in sorted(os.listdir(folder)):
        if file.endswith(".png"):
            st.image(
                os.path.join(folder, file),
                caption=file.replace("_", " ").replace(".png", ""),
                use_container_width=True
            )

# Tabs for sections
tabs = st.tabs(["📌 Univariate", "🔁 Bivariate", "🧩 Multivariate", "📎 Categorical Trends"])

with tabs[0]:
    st.markdown("### 📌 Univariate Analysis")
    show_images_from_folder("univariate", "Univariate Distributions")

with tabs[1]:
    st.markdown("### 🔁 Bivariate Analysis")
    show_images_from_folder("output_graphs/bivariate", "Output Graphs (Bivariate)")
    show_images_from_folder("bivariate_analysis", "Bivariate Analysis")

with tabs[2]:
    st.markdown("### 🧩 Multivariate Analysis")
    show_images_from_folder("multivariate", "Multivariate Trends")
    show_images_from_folder("output_graphs/multivariate", "Output Graphs (Multivariate)")

with tabs[3]:
    st.markdown("### 📎 Categorical Correlation & Premium Trends")
    show_images_from_folder("premium_trend_correlation_categorical", "Categorical Premium Trends")

# ======================================
# 🤖 PREMIUM PREDICTION SECTION
# ======================================
st.markdown("---")
st.title("💼 Insurance Premium Prediction")

st.markdown("🔄 Loading model, scaler, and top features...")
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
top_features = joblib.load("models/feature_order.pkl")
st.success("✅ Model, scaler, and features loaded.")

st.write("### Enter customer details below:")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        health_score = st.slider("Health Score", 0, 100, 75)
        age = st.slider("Age", 18, 100, 40)
        credit_score = st.slider("Credit Score", 300, 850, 680)
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0.0, max_value=30.0, value=3.0)

    with col2:
        annual_income = st.number_input("Annual Income (₦)", min_value=10000.0, value=5_000_000.0)
        insurance_duration = st.slider("Insurance Duration (years)", 0, 50, 3)
        dependents = st.slider("Number of Dependents", 0, 10, 1)
        prev_claims = st.slider("Previous Claims", 0, 5, 1)

    with col3:
        gender_male = st.selectbox("Gender", ["Male", "Female"]) == "Male"
        smoker_yes = st.selectbox("Smoking Status", ["Yes", "No"]) == "Yes"
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
        property_type = st.selectbox("Property Type", ["House", "Condo", "Apartment"])

    col4, col5, col6 = st.columns(3)
    with col4:
        policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    with col5:
        feedback = st.selectbox("Customer Feedback", ["Good", "Average", "Poor"])
    with col6:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    occupation_unknown = st.checkbox("Occupation Unknown")

    col7, col8 = st.columns(2)
    with col7:
        exercise_freq = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    with col8:
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

    submitted = st.form_submit_button("📈 Predict Premium")

# === Process Input and Predict ===
if submitted:
    income_log = np.log1p(annual_income)
    prev_claims_log = np.log1p(prev_claims)

    loc_suburban = location == "Suburban"
    loc_urban = location == "Urban"
    prop_condo = property_type == "Condo"
    prop_house = property_type == "House"
    policy_premium = policy_type == "Premium"
    policy_comprehensive = policy_type == "Comprehensive"
    feedback_poor = feedback == "Poor"
    feedback_good = feedback == "Good"
    ms_single = marital_status == "Single"
    ms_married = marital_status == "Married"
    ex_weekly = exercise_freq == "Weekly"
    ex_monthly = exercise_freq == "Monthly"
    ex_rarely = exercise_freq == "Rarely"
    edu_hs = education == "High School"
    edu_masters = education == "Master's"
    edu_phd = education == "PhD"

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
    }

    cleaned_input = {feat: input_data.get(feat, 0) for feat in top_features}
    ordered_df = pd.DataFrame([cleaned_input], columns=top_features)

    if hasattr(scaler, 'feature_names_in_'):
        missing = set(scaler.feature_names_in_) - set(ordered_df.columns)
        extra = set(ordered_df.columns) - set(scaler.feature_names_in_)
        if missing or extra:
            st.error(f"❌ Feature mismatch. Missing: {missing}, Extra: {extra}")
    try:
        scaled_input = scaler.transform(ordered_df)
        st.write("🔢 Scaled Input Used for Prediction:")
        st.dataframe(pd.DataFrame(scaled_input, columns=top_features))

        raw_output = model.predict(scaled_input)
        prediction = max(0, raw_output[0])
        st.success(f"💡 **Predicted Premium Amount:** ₦{prediction:,.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

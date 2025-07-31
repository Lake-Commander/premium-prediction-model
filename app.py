import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

# Manual mappings (must match training-time encoding)
gender_map = {"Male": 0, "Female": 1}
city_map = {"Urban": 0, "Rural": 1}
region_map = {"North": 0, "South": 1, "East": 2, "West": 3}
residence_map = {"Owned": 0, "Rented": 1}
policy_channel_map = {"Online": 0, "Agent": 1, "Branch": 2}

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("🏷️ Insurance Premium Predictor")
st.markdown("Estimate your premium based on customer & policy info.")

with st.form("predict_form"):
    st.subheader("Enter Details")

    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", list(gender_map.keys()))
    city = st.selectbox("City Type", list(city_map.keys()))
    region = st.selectbox("Region", list(region_map.keys()))
    residence = st.selectbox("Residence Type", list(residence_map.keys()))
    dependents = st.slider("Number of Dependents", 0, 10, 1)
    income = st.number_input("Annual Income (USD)", 1000, 1_000_000, 30000, step=500)
    policy_channel = st.selectbox("Policy Sales Channel", list(policy_channel_map.keys()))
    tenure = st.slider("Policy Tenure (years)", 1, 30, 5)
    no_of_policies = st.slider("Previous Policies", 0, 10, 1)

    submitted = st.form_submit_button("Predict Premium")

if submitted:
    # Encode inputs manually
    input_df = pd.DataFrame([[
        age,
        gender_map[gender],
        city_map[city],
        region_map[region],
        residence_map[residence],
        dependents,
        income,
        policy_channel_map[policy_channel],
        tenure,
        no_of_policies
    ]], columns=[
        "Age", "Gender", "City Type", "Region", "Type of Residence",
        "Number of Dependents", "Annual Income", "Policy Sales Channel",
        "Policy Tenure", "Number of Policies"
    ])

    # Predict
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Premium: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

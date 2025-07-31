import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load data
data = pd.read_csv("Insurance Premium Prediction Dataset.csv")

# Load model (optional scaler)
model = joblib.load("random_forest_model.pkl")
try:
    scaler = joblib.load("scaler.pkl") # None exist here though
    scale_input = True
except:
    scaler = None
    scale_input = False

st.title("💼 Insurance Premium Estimator")
st.markdown("Fill in the details below to get a premium estimate.")

# Start form
with st.form("input_form"):
    age = st.slider("Age", int(data["Age"].min()), int(data["Age"].max()), 30)
    gender = st.selectbox("Gender", sorted(data["Gender"].dropna().unique()))
    income = st.slider("Annual Income", int(data["Annual Income"].min()), int(data["Annual Income"].max()), 50000)
    marital_status = st.selectbox("Marital Status", sorted(data["Marital Status"].dropna().unique()))
    dependents = st.slider("Number of Dependents", 0, int(data["Number of Dependents"].max()), 1)
    education = st.selectbox("Education Level", sorted(data["Education Level"].dropna().unique()))
    occupation = st.selectbox("Occupation", sorted(data["Occupation"].dropna().unique()))
    health_score = st.slider("Health Score", int(data["Health Score"].min()), int(data["Health Score"].max()), 50)
    location = st.selectbox("Location", sorted(data["Location"].dropna().unique()))
    policy_type = st.selectbox("Policy Type", sorted(data["Policy Type"].dropna().unique()))
    prev_claims = st.slider("Previous Claims", 0, int(data["Previous Claims"].max()), 0)
    vehicle_age = st.slider("Vehicle Age", 0, int(data["Vehicle Age"].max()), 3)
    credit_score = st.slider("Credit Score", int(data["Credit Score"].min()), int(data["Credit Score"].max()), 600)
    duration = st.slider("Insurance Duration", 1, int(data["Insurance Duration"].max()), 5)
    feedback = st.selectbox("Customer Feedback", sorted(data["Customer Feedback"].dropna().unique()))
    smoking_status = st.selectbox("Smoking Status", sorted(data["Smoking Status"].dropna().unique()))
    exercise = st.selectbox("Exercise Frequency", sorted(data["Exercise Frequency"].dropna().unique()))
    property_type = st.selectbox("Property Type", sorted(data["Property Type"].dropna().unique()))

    submitted = st.form_submit_button("Predict")

# When form is submitted
if submitted:
    input_dict = {
        "Age": age,
        "Gender": gender,
        "Annual Income": income,
        "Marital Status": marital_status,
        "Number of Dependents": dependents,
        "Education Level": education,
        "Occupation": occupation,
        "Health Score": health_score,
        "Location": location,
        "Policy Type": policy_type,
        "Previous Claims": prev_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": duration,
        "Customer Feedback": feedback,
        "Smoking Status": smoking_status,
        "Exercise Frequency": exercise,
        "Property Type": property_type
    }

    # Create dataframe from input
    input_df = pd.DataFrame([input_dict])

    # Handle encoding (must match model's training)
    full_df = pd.concat([input_df, data], ignore_index=True)
    input_encoded = pd.get_dummies(full_df, drop_first=True)
    input_encoded = input_encoded.iloc[0:1]  # only the first row (our input)

    # Reorder columns to match model input
    model_features = model.feature_names_in_
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    if scale_input and scaler:
        input_encoded = scaler.transform(input_encoded)

    prediction = model.predict(input_encoded)[0]

    st.success(f"💰 Estimated Premium: **₦{prediction:,.2f}**")

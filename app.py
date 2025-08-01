import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Set title
st.title("Insurance Premium Prediction")
st.subheader("Enter Customer Information")

# Define input fields
input_data = {
    'Health Score': st.number_input("Health Score", value=0.0),
    'Age': st.number_input("Age", value=0.0),
    'Credit Score': st.number_input("Credit Score", value=0.0),
    'Vehicle Age': st.number_input("Vehicle Age", value=0.0),
    'Annual Income_log': st.number_input("Annual Income_log", value=0.0),
    'Annual Income': st.number_input("Annual Income", value=0.0),
    'Insurance Duration': st.number_input("Insurance Duration", value=0.0),
    'Number of Dependents': st.number_input("Number of Dependents", value=0.0),
    'Previous Claims': st.number_input("Previous Claims", value=0.0),
    'Previous Claims_log': st.number_input("Previous Claims_log", value=0.0),
    'Gender_Male': st.number_input("Gender_Male", value=0.0),
    'Smoking Status_Yes': st.number_input("Smoking Status_Yes", value=0.0),
    'Location_Suburban': st.number_input("Location_Suburban", value=0.0),
    'Property Type_Condo': st.number_input("Property Type_Condo", value=0.0),
    'Location_Urban': st.number_input("Location_Urban", value=0.0),
    'Policy Type_Premium': st.number_input("Policy Type_Premium", value=0.0),
    'Customer Feedback_Poor': st.number_input("Customer Feedback_Poor", value=0.0),
    'Marital Status_Single': st.number_input("Marital Status_Single", value=0.0),
    'Property Type_House': st.number_input("Property Type_House", value=0.0),
    'Occupation_Unknown': st.number_input("Occupation_Unknown", value=0.0),
    'Marital Status_Married': st.number_input("Marital Status_Married", value=0.0),
    'Exercise Frequency_Monthly': st.number_input("Exercise Frequency_Monthly", value=0.0),
    'Exercise Frequency_Rarely': st.number_input("Exercise Frequency_Rarely", value=0.0),
    'Education Level_PhD': st.number_input("Education Level_PhD", value=0.0),
    'Customer Feedback_Good': st.number_input("Customer Feedback_Good", value=0.0),
    'Policy Type_Comprehensive': st.number_input("Policy Type_Comprehensive", value=0.0),
    "Education Level_Master's": st.number_input("Education Level_Master's", value=0.0),
    'Exercise Frequency_Weekly': st.number_input("Exercise Frequency_Weekly", value=0.0),
    'Education Level_High School': st.number_input("Education Level_High School", value=0.0)
}

# Convert to array and reshape
input_array = np.array([list(input_data.values())])
scaled_input = scaler.transform(input_array)

# Predict
if st.button("Predict Premium"):
    pred_log = model.predict(scaled_input)[0]

    # Apply clipping to prevent negative predictions
    pred_log = max(pred_log, 0)  # clip at zero log
    pred = max(np.expm1(pred_log), 1000)  # apply inverse log and floor of ₦1000

    st.success(f"Estimated Premium: ₦{pred:,.2f}")

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Define feature columns (based on your training)
FEATURES = [
    "age", "sex", "bmi", "children", "smoker", "region"
]

# Preprocessing function
def preprocess_input(input_df):
    df = input_df.copy()
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df)
    
    # Ensure all expected columns are present
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match model's training
    df = df[model.feature_names_in_]
    return df

# Streamlit UI
st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("💰 Insurance Premium Prediction App")

st.markdown("Enter the details below to estimate the insurance premium.")

# Input fields
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Create DataFrame from user input
input_data = {
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict Premium"):
    try:
        input_processed = preprocess_input(input_df)
        prediction = model.predict(input_processed)[0]
        st.success(f"🏷️ Estimated Premium Amount: ₦{prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

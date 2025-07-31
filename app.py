import streamlit as st
import pandas as pd
import joblib

# Load dataset to extract options
df = pd.read_csv("Insurance Premium Prediction Dataset.csv")

# Load model and optionally a scaler if used
model = joblib.load("random_forest_model.pkl")

# Optional scaler
try:
    scaler = joblib.load("models/scaler.pkl")
    use_scaler = True
except:
    scaler = None
    use_scaler = False

# Get dynamic options from the dataset
def get_unique_sorted(column):
    values = df[column].dropna().unique()
    return sorted(values)

# App title
st.title("🧮 Insurance Premium Predictor")
st.markdown("Enter client details to estimate the insurance premium.")

with st.form("input_form"):
    age = st.slider("Age", 18, int(df["Age"].max()))
    gender = st.selectbox("Gender", get_unique_sorted("Gender"))
    income = st.slider("Annual Income (₦)", 10000, int(df["Annual Income"].max()))
    marital = st.selectbox("Marital Status", get_unique_sorted("Marital Status"))
    dependents = st.slider("Number of Dependents", 0, int(df["Number of Dependents"].max()))
    education = st.selectbox("Education Level", get_unique_sorted("Education Level"))
    occupation = st.selectbox("Occupation", get_unique_sorted("Occupation"))
    health = st.slider("Health Score", 1, int(df["Health Score"].max()))
    location = st.selectbox("Location", get_unique_sorted("Location"))
    policy_type = st.selectbox("Policy Type", get_unique_sorted("Policy Type"))
    prev_claims = st.slider("Previous Claims", 0, int(df["Previous Claims"].max()))
    vehicle_age = st.slider("Vehicle Age", 0, int(df["Vehicle Age"].max()))
    credit_score = st.slider("Credit Score", 0, int(df["Credit Score"].max()))
    insurance_duration = st.slider("Insurance Duration (years)", 0, int(df["Insurance Duration"].max()))
    feedback = st.selectbox("Customer Feedback", get_unique_sorted("Customer Feedback"))
    smoking = st.selectbox("Smoking Status", get_unique_sorted("Smoking Status"))
    exercise = st.selectbox("Exercise Frequency", get_unique_sorted("Exercise Frequency"))
    property_type = st.selectbox("Property Type", get_unique_sorted("Property Type"))

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "Age": age,
        "Gender": gender,
        "Annual Income": income,
        "Marital Status": marital,
        "Number of Dependents": dependents,
        "Education Level": education,
        "Occupation": occupation,
        "Health Score": health,
        "Location": location,
        "Policy Type": policy_type,
        "Previous Claims": prev_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": insurance_duration,
        "Customer Feedback": feedback,
        "Smoking Status": smoking,
        "Exercise Frequency": exercise,
        "Property Type": property_type
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categoricals like training
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = input_df[col].astype('category')
        if col in df.columns and df[col].dtype.name == 'category':
            input_df[col] = input_df[col].cat.set_categories(df[col].unique())

    # Get dummies to match training format
    input_df = pd.get_dummies(input_df)
    df_encoded = pd.get_dummies(df[input_df.columns], drop_first=False)
    input_df = input_df.reindex(columns=df_encoded.columns, fill_value=0)

    # Scale if scaler exists
    if use_scaler:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df

    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Estimated Premium: **₦{prediction:,.2f}**")

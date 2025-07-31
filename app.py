import streamlit as st
import pandas as pd
import joblib

# --- CACHE DATA LOADING ---
@st.cache_data
def load_data():
    columns_needed = [
        'Age', 'Gender', 'Annual Income', 'Marital Status',
        'Education Level', 'Occupation', 'Location',
        'Property Type', 'Policy Type'
    ]
    return pd.read_csv("Insurance Premium Prediction Dataset.csv", usecols=columns_needed)

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")  # or your model's filename

# --- UI CONFIG ---
st.set_page_config(page_title="Premium Predictor", layout="centered")
st.title("💰 Insurance Premium Predictor")
st.markdown("Enter client details below to predict the insurance premium.")

# --- Load Data and Model ---
df = load_data()
model = load_model()

# --- Cached Dropdown Values ---
@st.cache_data
def get_dropdowns(df):
    return {
        "gender": df['Gender'].dropna().unique().tolist(),
        "marital": df['Marital Status'].dropna().unique().tolist(),
        "education": df['Education Level'].dropna().unique().tolist(),
        "occupation": df['Occupation'].dropna().unique().tolist(),
        "location": df['Location'].dropna().unique().tolist(),
        "property_type": df['Property Type'].dropna().unique().tolist(),
        "policy_type": df['Policy Type'].dropna().unique().tolist()
    }

dropdowns = get_dropdowns(df)

# --- Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 35)
        gender = st.selectbox("Gender", dropdowns['gender'])
        income = st.number_input("Annual Income (USD)", min_value=1000, value=50000)
        marital_status = st.selectbox("Marital Status", dropdowns['marital'])
        education = st.selectbox("Education Level", dropdowns['education'])

    with col2:
        occupation = st.selectbox("Occupation", dropdowns['occupation'])
        location = st.selectbox("Location", dropdowns['location'])
        property_type = st.selectbox("Property Type", dropdowns['property_type'])
        policy_type = st.selectbox("Policy Type", dropdowns['policy_type'])

    submitted = st.form_submit_button("Predict Premium")

# --- Prediction ---
if submitted:
    try:
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Annual Income': income,
            'Marital Status': marital_status,
            'Education Level': education,
            'Occupation': occupation,
            'Location': location,
            'Property Type': property_type,
            'Policy Type': policy_type
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"💸 Predicted Insurance Premium: **${prediction:,.2f}**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

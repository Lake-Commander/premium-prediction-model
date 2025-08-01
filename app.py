import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()

# Expected features from the model
expected_features = list(model.feature_names_in_)

st.title("💸 Insurance Premium Prediction App")
st.markdown("Enter values for each feature to predict the insurance premium.")

# Create input fields dynamically
user_input = {}
for feature in expected_features:
    if "log" in feature.lower() or "score" in feature.lower() or "Amount" in feature:
        val = st.number_input(f"{feature}", format="%.4f")
    else:
        val = st.number_input(f"{feature}", step=1.0)
    user_input[feature] = val

# Prediction button
if st.button("Predict Premium"):
    try:
        # Create DataFrame from user input
        X = pd.DataFrame([user_input])

        # Check for missing or extra features
        missing = set(expected_features) - set(X.columns)
        extra = set(X.columns) - set(expected_features)

        if missing:
            st.error(f"❌ Missing features: {missing}")
        elif extra:
            st.warning(f"⚠️ Extra features provided: {extra}")
        else:
            # Reorder columns to match training
            X = X[expected_features]

            # Predict and convert from log scale if applicable
            pred_log = model.predict(X)[0]
            pred_actual = np.exp(pred_log)

            st.success(f"💰 Predicted Insurance Premium: **${pred_actual:,.2f}**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

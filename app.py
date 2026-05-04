import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("waste_model.pkl")
le_desc = joblib.load("label_desc.pkl")
le_mat = joblib.load("label_mat.pkl")

st.title("Waste Classification App")

# Inputs
desc = st.text_input("Description (e.g. plastic bottle)")
material = st.selectbox("Material", ["plastic", "organic", "metal", "paper"])

weight = st.number_input("Weight (grams)", min_value=0.0)
moisture = st.number_input("Moisture Level", min_value=0.0)
reusability = st.number_input("Reusability Score", min_value=0.0)

# Predict
if st.button("Predict"):
    try:
        # Encode text inputs
        desc_enc = le_desc.transform([desc])[0]
        mat_enc = le_mat.transform([material])[0]

        # Combine ALL 5 features
        features = np.array([[desc_enc, mat_enc, weight, moisture, reusability]])

        prediction = model.predict(features)

        st.success(f"Predicted Waste Type: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")

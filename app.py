# app.py

import streamlit as st
import joblib
import numpy as np

# =====================
# LOAD SAVED FILES
# =====================
model = joblib.load("waste_model.pkl")
le_mat = joblib.load("label_mat.pkl")
le_target = joblib.load("label_target.pkl")
scaler = joblib.load("scaler.pkl")

# =====================
# TITLE
# =====================
st.title("♻️ Waste Classification System")

# =====================
# USER INPUTS
# =====================
material = st.selectbox("Select Material", le_mat.classes_)

weight = st.number_input("Weight (grams)", min_value=0.0, step=1.0)

moisture = st.slider("Moisture Level", 0.0, 1.0)

reusability = st.slider("Reusability Score", 0.0, 1.0)

# =====================
# PREDICTION
# =====================
if st.button("Predict Waste Type"):

    try:
        # Encode material
       material = st.selectbox("Select Material", le_mat.classes_)

        # Create input array
        sample = np.array([[material_encoded, weight, moisture, reusability]])

        # Apply scaling
        sample = scaler.transform(sample)

        # Predict
        prediction = model.predict(sample)

        # Convert numeric → text
        result = le_target.inverse_transform(prediction)[0]

        # Display result
        st.success(f"♻️ Waste Type: {result}")

    except Exception as e:
        st.error(f"Error: {e}")

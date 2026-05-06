# app.py

import streamlit as st
import joblib
import numpy as np

# =====================
# LOAD MODELS & ENCODERS
# =====================
model = joblib.load("waste_model.pkl")
le_mat = joblib.load("label_mat.pkl")
le_target = joblib.load("label_target.pkl")
scaler = joblib.load("scaler.pkl")

# =====================
# APP TITLE
# =====================
st.title("♻️ Waste Classification System")

st.write("Predict waste type based on material, weight, moisture, and reusability.")

# =====================
# USER INPUTS (TEXTUAL UI)
# =====================
material = st.selectbox("Select Material", le_mat.classes_)
weight = st.number_input("Weight (grams)", min_value=0.0, step=1.0)
moisture = st.slider("Moisture Level", 0.0, 1.0, step=0.01)
reusability = st.slider("Reusability Score", 0.0, 1.0, step=0.01)

# =====================
# PREDICTION BUTTON
# =====================
if st.button("Predict Waste Type"):

    try:
        # =====================
        # ENCODE MATERIAL (TEXT → NUMBER)
        # =====================
        material_encoded = le_mat.transform([material])[0]

        # =====================
        # CREATE INPUT ARRAY
        # =====================
        sample = np.array([[material_encoded, weight, moisture, reusability]])

        # =====================
        # SCALE INPUT
        # =====================
        sample_scaled = scaler.transform(sample)

        # =====================
        # PREDICT
        # =====================
        prediction = model.predict(sample_scaled)

        # =====================
        # CONVERT BACK TO TEXT LABEL
        # =====================
        result = le_target.inverse_transform(prediction)[0]

        # =====================
        # DISPLAY RESULT
        # =====================
        st.success(f"♻️ Predicted Waste Type: {result}")

    except Exception as e:
        st.error(f"Error occurred: {e}")

import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("waste_model.pkl")
le_desc = joblib.load("label_desc.pkl")
le_mat = joblib.load("label_mat.pkl")
le_target = joblib.load("label_target.pkl")  # ✅ NEW

st.title("Waste Classification App")

# Inputs
desc = st.text_input("Description (e.g. plastic bottle)")
material = st.selectbox("Material", ["plastic", "organic", "metal", "paper"])

weight = st.number_input("Weight (grams)", min_value=0.0)
moisture = st.number_input("Moisture Level (0–1)", min_value=0.0, max_value=1.0)
reusability = st.number_input("Reusability Score (0–1)", min_value=0.0, max_value=1.0)

# Predict
if st.button("Predict"):
    try:
        # Encode inputs
        desc_enc = le_desc.transform([desc])[0]
        mat_enc = le_mat.transform([material])[0]

        # Combine features
        features = np.array([[desc_enc, mat_enc, weight, moisture, reusability]])

        # Predict
        prediction = model.predict(features)

        # ✅ Decode result
        predicted_label = le_target.inverse_transform(prediction)[0]

        st.success(f"Predicted Waste Type: {predicted_label}")

    except ValueError as ve:
        st.error("⚠️ Input error: Make sure your description exists in training data.")
        st.error(str(ve))

    except Exception as e:
        st.error(f"Error: {e}")

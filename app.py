import streamlit as st
import pickle
import numpy as np

# Load model
with open("waste_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Waste Classification App")

st.write("Enter waste details:")

# Inputs
weight = st.number_input("Weight (grams)", min_value=0.0)
moisture = st.number_input("Moisture Level", min_value=0.0)
reusability = st.number_input("Reusability Score", min_value=0.0)

# Predict
if st.button("Predict"):
    features = np.array([[weight, moisture, reusability]])
    prediction = model.predict(features)
    st.success(f"Predicted Waste Type: {prediction[0]}")

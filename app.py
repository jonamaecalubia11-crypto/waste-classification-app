import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =====================
# MODEL SETUP
# =====================
@st.cache_resource
def setup_model():
    data = {
        "description": [
            "banana peel","leftover rice","vegetable scraps","egg shells","coffee grounds",
            "fruit peel","spoiled bread","used tissue","wet cardboard","food waste mix",
            "plastic bottle","glass bottle","aluminum can","tin can","newspaper",
            "magazine","clean cardboard","steel scrap","plastic container","glass jar",
            "candy wrapper","chip bag","styrofoam box","dirty plastic cup","sachet packaging",
            "broken ceramic plate","old sponge","rubber scraps","diaper","sanitary waste"
        ],
        "material": [
            "organic","organic","organic","organic","organic",
            "organic","organic","paper","paper","organic",
            "plastic","glass","metal","metal","paper",
            "paper","paper","metal","plastic","glass",
            "plastic","plastic","plastic","plastic","plastic",
            "glass","plastic","plastic","plastic","plastic"
        ],
        "weight_grams": [120,200,180,70,90,110,130,25,140,300,45,250,30,55,40,60,150,500,35,220,10,15,40,30,5,320,50,80,200,150],
        "moisture_level": [0.92,0.95,0.88,0.80,0.85,0.90,0.75,0.70,0.65,0.96,0.05,0.00,0.00,0.00,0.20,0.15,0.10,0.00,0.05,0.00,0.00,0.00,0.00,0.30,0.00,0.00,0.20,0.10,0.70,0.60],
        "reusability_score": [0.05,0.00,0.05,0.10,0.00,0.05,0.02,0.05,0.10,0.00,0.85,0.95,0.90,0.85,0.75,0.70,0.80,0.95,0.80,0.90,0.20,0.15,0.10,0.20,0.10,0.05,0.05,0.10,0.00,0.00],
        "waste_type": [
            "biodegradable","biodegradable","biodegradable","biodegradable","biodegradable",
            "biodegradable","biodegradable","biodegradable","biodegradable","biodegradable",
            "recyclable","recyclable","recyclable","recyclable","recyclable",
            "recyclable","recyclable","recyclable","recyclable","recyclable",
            "residual","residual","residual","residual","residual",
            "residual","residual","residual","residual","residual"
        ]
    }

    df = pd.DataFrame(data)

    le_desc = LabelEncoder()
    le_mat = LabelEncoder()
    le_target = LabelEncoder()

    df_train = df.copy()

    df_train["description"] = le_desc.fit_transform(df_train["description"])
    df_train["material"] = le_mat.fit_transform(df_train["material"])
    df_train["waste_type"] = le_target.fit_transform(df_train["waste_type"])

    X = df_train.drop("waste_type", axis=1)
    y = df_train["waste_type"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, le_desc, le_mat, le_target


model, scaler, le_desc, le_mat, le_target = setup_model()

# =====================
# UI
# =====================
st.title("♻️ Waste Classification System")
st.write("Enter waste details to predict its type.")

# ✅ CHANGED: Manual text input instead of selectbox
description = st.text_input("Enter Item Description (e.g. banana peel)")
material = st.selectbox("Material", le_mat.classes_)

weight = st.number_input("Weight (grams)", value=100.0)
moisture = st.slider("Moisture Level", 0.0, 1.0, 0.5)
reusability = st.slider("Reusability Score", 0.0, 1.0, 0.1)

# =====================
# PREDICTION
# =====================
if st.button("Predict Waste Type"):

    # Validate description input
    if description.strip() == "":
        st.error("Please enter a valid description.")
    elif description not in le_desc.classes_:
        st.error("Unknown description. Try using training data words like 'banana peel', 'plastic bottle', etc.")
    else:
        desc_encoded = le_desc.transform([description])[0]
        mat_encoded = le_mat.transform([material])[0]

        sample = np.array([[desc_encoded, mat_encoded, weight, moisture, reusability]])
        sample_scaled = scaler.transform(sample)

        prediction = model.predict(sample_scaled)[0]
        result = le_target.inverse_transform([prediction])[0]

        st.success(f"♻️ Predicted Waste Type: {result.upper()}")

        if result == "biodegradable":
            st.info("Can be composted.")
        elif result == "recyclable":
            st.info("Send to recycling center.")
        else:
            st.warning("Dispose as general waste.")


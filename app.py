import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- SETUP (Required for the button to work) ---
# This section recreates the model and tools from test_waste.ipynb[cite: 1]
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
    df = pd.DataFrame(data)[cite: 1]
    
    le_desc = LabelEncoder()
    le_mat = LabelEncoder()
    le_target = LabelEncoder()
    
    # Pre-process features for training[cite: 1]
    df_train = df.copy()
    df_train["description"] = le_desc.fit_transform(df_train["description"])
    df_train["material"] = le_mat.fit_transform(df_train["material"])
    df_train["waste_type"] = le_target.fit_transform(df_train["waste_type"])
    
    X = df_train.drop("waste_type", axis=1)[cite: 1]
    y = df_train["waste_type"][cite: 1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)[cite: 1]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)[cite: 1]
    
    return model, scaler, le_desc, le_mat, le_target

model, scaler, le_desc, le_mat, le_target = setup_model()

# --- INPUT UI ---
st.title("Waste Classifier")
item_desc = st.selectbox("Description", le_desc.classes_)
item_material = st.selectbox("Material", le_mat.classes_)
weight = st.number_input("Weight (grams)", value=100)
moisture = st.slider("Moisture Level", 0.0, 1.0, 0.1)
reusability = st.slider("Reusability Score", 0.0, 1.0, 0.1)

# --- YOUR REQUESTED BLOCK ---
if st.button("Classify Waste"):
    # Encode inputs
    desc_encoded = le_desc.transform([item_desc])[0]
    mat_encoded = le_mat.transform([item_material])[0]
    
    # Create feature array
    features = np.array([[desc_encoded, mat_encoded, weight, moisture, reusability]])
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction_idx = model.predict(features_scaled)[0]
    prediction_label = le_target.inverse_transform([prediction_idx])[0]
    
    # Display Results
    st.divider()
    st.subheader(f"Result: {prediction_label.upper()}")
    
    if prediction_label == "biodegradable":
        st.success("This item can be composted!")
    elif prediction_label == "recyclable":
        st.info("Ensure this item is clean before recycling.")
    else:
        st.warning("This item should be disposed of in general waste.")

# Display training data overview
if st.checkbox("Show Training Data Sample"):
    st.write("This is a preview of the data used to train the model:")
    # Reconstructing a readable dataframe for display
    display_df = pd.DataFrame({
        "Description": le_desc.classes_,
        "Type": ["Refer to training set" for _ in le_desc.classes_]
    })
    st.dataframe(display_df.head())

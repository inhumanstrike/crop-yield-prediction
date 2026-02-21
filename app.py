# =====================================
# 🌾 Final Clean Streamlit App
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained pipeline
model = joblib.load("models/crop_yield_pipeline.pkl")

# Load dataset for dropdown options
df = pd.read_csv("Final_Dataset_after_temperature.csv")

st.title("🌾 Intelligent Crop Yield Prediction System")

st.write("Select agricultural parameters below.")

# Dropdown inputs
state = st.selectbox("State", sorted(df["State_Name"].unique()))
crop_type = st.selectbox("Crop Type", sorted(df["Crop_Type"].unique()))
crop = st.selectbox("Crop", sorted(df["Crop"].unique()))

rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=1000.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, value=25.0)
area = st.number_input("Area in hectares", min_value=0.0, value=10.0)

if st.button("Predict Yield"):

    input_data = pd.DataFrame([{
        "State_Name": state,
        "Crop_Type": crop_type,
        "Crop": crop,
        "rainfall": rainfall,
        "temperature": temperature,
        "Area_in_hectares": area,
        "Rainfall_Temp": rainfall * temperature,
        "Rainfall_sq": rainfall ** 2,
        "Temp_sq": temperature ** 2,
        "Area_log": np.log1p(area)
    }])

    # Predict (log scale)
    pred_log = model.predict(input_data)[0]

    # Convert back from log scale
    prediction = np.expm1(pred_log)

    st.success(f"🌱 Predicted Yield: {round(prediction, 2)} tons per hectare")
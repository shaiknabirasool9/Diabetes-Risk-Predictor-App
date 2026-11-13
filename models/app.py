import joblib
import numpy as np
import streamlit as st
from pathlib import Path

# Load the saved model and scaler artifacts
MODEL_DIR = Path(__file__).resolve().parent
model = joblib.load(MODEL_DIR / "diabetes_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

# App Layout & Title
st.set_page_config(
    page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered"
)
st.title("🩺 Real-Time Diabetes Risk Predictor")
st.write(
    "Enter the patient's health parameters below to receive an instant"
    " prediction from your trained model."
)

# Input Fields Organized in Columns
col1, col2 = st.columns(2)

with col1:
  pregnancies = st.number_input(
      "Pregnancies", min_value=0, max_value=20, value=1, step=1
  )
  glucose = st.number_input(
      "Glucose", min_value=0.0, max_value=300.0, value=112.0
  )
  blood_pressure = st.number_input(
      "Blood Pressure", min_value=0.0, max_value=200.0, value=72.0
  )
  skin_thickness = st.number_input(
      "Skin Thickness", min_value=0.0, max_value=100.0, value=25.0
  )

with col2:
  insulin = st.number_input(
      "Insulin", min_value=0.0, max_value=900.0, value=104.0
  )
  bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=29.1)
  dpf = st.number_input(
      "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.43
  )
  age = st.number_input("Age", min_value=1, max_value=120, value=42, step=1)

# Prediction Button & Logic
if st.button("Predict Diabetes Status"):
  # Create input array matching the 8 feature columns
  input_data = np.array([[
      pregnancies,
      glucose,
      blood_pressure,
      skin_thickness,
      insulin,
      bmi,
      dpf,
      age,
  ]])

  # Scale data using the saved StandardScaler
  scaled_data = scaler.transform(input_data)

  # Run prediction and probability calculation
  prediction = model.predict(scaled_data)[0]
  probability = model.predict_proba(scaled_data)[0][1]

  st.subheader("Prediction Results:")
  if prediction == 1:
    st.error(
        f"⚠️ The model predicts the patient is **Diabetic** (Confidence:"
        f" {probability:.2%})"
    )
  else:
    st.success(
        f"✅ The model predicts the patient is **Non-Diabetic** (Confidence:"
        f" {(1 - probability):.2%})"
    )
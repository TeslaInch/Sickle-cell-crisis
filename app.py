import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open('model/crisis_model.pkl', 'rb'))

st.title("🩺 Sickle Cell Crisis Predictor")

st.subheader("Enter Your Health Data")
pain = st.slider("Pain Level (1-10)", 1, 10, 5)
hydration = st.number_input("Hydration (Liters)", 0.5, 5.0, 2.0, step=0.1)
medication = st.radio("Medication Taken?", ["Yes", "No"])
temperature = st.number_input("Body Temperature (°C)", 35.0, 42.0, 37.0, step=0.1)
fatigue = st.slider("Fatigue Level (1-10)", 1, 10, 5)

if st.button("Predict Crisis Risk"):
    features = [[
        pain,
        hydration,
        1 if medication == "Yes" else 0,
        temperature,
        fatigue
    ]]
    prediction = model.predict(features)[0]
    risk = "⚠️ High Risk of Crisis" if prediction == 1 else "✅ Low Risk"
    st.info(f"Prediction: **{risk}**")

# -------------------------------
# Disclaimer Section
# -------------------------------
st.markdown("---")
st.markdown(
    """
    ⚠️ **Disclaimer**  
    This app is **only a support tool** to help sickle cell patients manage their health.  
    It **does not provide medical advice** and should **not replace consultation with qualified healthcare professionals**.  
    Always seek immediate medical attention during a health crisis.
    """,
    unsafe_allow_html=True
)

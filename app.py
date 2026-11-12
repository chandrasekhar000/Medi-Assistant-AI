
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Symptom Checker",
    page_icon="🏥",
    layout="wide"
)

# Load trained model
try:
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
except FileNotFoundError:
    st.error("Model files not found. Run train_model.py first.")
    st.stop()

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 25px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🏥 AI-Based Medical Symptom Checker")
st.write("An educational machine-learning demonstration")

st.info(
    "This application is for educational purposes only. "
    "It cannot diagnose diseases or replace medical advice."
)

st.divider()

# Symptom input
st.subheader("Select Your Symptoms")

symptoms = {}

col1, col2, col3 = st.columns(3)

columns = [col1, col2, col3]

for index, feature in enumerate(features):
    with columns[index % 3]:
        symptoms[feature] = st.checkbox(
            feature.replace("_", " ").title()
        )

st.divider()

# Prediction
if st.button("Analyze Symptoms"):

    input_data = pd.DataFrame(
        [[int(symptoms[f]) for f in features]],
        columns=features
    )

    if not any(symptoms.values()):
        st.warning("Please select at least one symptom.")

    else:
        prediction = model.predict(input_data)[0]

        st.subheader("Model Output")

        st.success(f"Sample model prediction: {prediction}")

        st.warning(
            "This result comes from a tiny synthetic dataset. "
            "It is not a medical assessment. Do not use it to "
            "make treatment decisions."
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            st.subheader("Model Scores")

            for condition, score in zip(
                model.classes_,
                probabilities
            ):
                st.write(f"{condition}: {score * 100:.2f}%")
                st.progress(float(score))

st.divider()

st.caption(
    "AI Medical Symptom Checker | Academic Project | "
    "Not intended for clinical use"
)

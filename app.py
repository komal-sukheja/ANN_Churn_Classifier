import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# -----------------------------
# Load Model & Transformers
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)

model = load_model()
label_encoder_gender = load_pickle("label_encoder_gender.pkl")
OHE_geography = load_pickle("OHE_geography.pkl")
scaler = load_pickle("scaler.pkl")

# -----------------------------
# Title + Description
# -----------------------------
st.title("📉 Customer Churn Prediction App")

st.write("""
This app predicts the **likelihood of customer churn** based on  customer profile, activity, and financial information.
""")

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("📝 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", OHE_geography.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", 18, 92)
    tenure = st.slider("Tenure (years)", 0, 10)
    num_of_products = st.slider("Number of Products", 1, 4)

with col2:
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000)
    balance = st.number_input("Balance", min_value=0.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active_member = st.selectbox("Active Member", [0, 1])

st.divider()

# -----------------------------
# Prepare Input
# -----------------------------
def preprocess():
    base = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    })

    geo_encoded = OHE_geography.transform([[geography]])
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=OHE_geography.get_feature_names_out(["Geography"])
    )

    final = pd.concat([base.reset_index(drop=True), geo_df], axis=1)
    return scaler.transform(final)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Churn"):
    data_scaled = preprocess()
    prediction = model.predict(data_scaled)
    prob = float(prediction[0][0])

    st.subheader("📊 Churn Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2f}")

    if prob > 0.5:
        st.error("❌ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")

st.divider()

# -----------------------------
# Footer
# -----------------------------
st.write("**Model:** Artificial Neural Network (ANN)")
st.write("**Tools:** TensorFlow, Scikit-learn, Streamlit")

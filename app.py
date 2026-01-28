import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
model = pickle.load(open("customer_model.pkl", "rb"))

st.set_page_config(page_title="Customer Purchase Prediction")

st.title("🛒 Customer Purchase Prediction")
st.write("Predict whether a customer will purchase based on input details")

# User inputs
age = st.number_input("Age", 18, 70, 30)

gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Female" else 0

income = st.number_input("Monthly Income", 10000, 100000, 40000)

score = st.slider("Spending Score", 0, 100, 50)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[age, gender, income, score]],
        columns=["Age", "Gender", "Monthly_Income", "Spending_Score"]
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Customer WILL Purchase")
    else:
        st.error("❌ Customer will NOT Purchase")

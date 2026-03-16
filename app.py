import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Customer Purchase Prediction")

@st.cache_resource
def load_model():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "customer_behavior_data.csv")
    df = pd.read_csv(csv_path)

    # Encode Gender if it's string
    if df['Gender'].dtype == object:
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

    # Find target column
    target_col = next((c for c in df.columns if 'purchase' in c.lower()), None)
    if target_col is None:
        raise ValueError("No purchase column found")

    feature_cols = ["Age", "Gender", "Monthly_Income", "Spending_Score"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[target_col]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, feature_cols

st.title("🛒 Customer Purchase Prediction")
st.write("Predict whether a customer will purchase based on input details")

model, feature_cols = load_model()

age    = st.number_input("Age", 18, 70, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Female" else 0
income = st.number_input("Monthly Income", 10000, 100000, 40000)
score  = st.slider("Spending Score", 0, 100, 50)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[age, gender, income, score]],
        columns=["Age", "Gender", "Monthly_Income", "Spending_Score"]
    )
    input_data = input_data[feature_cols]
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1] * 100

    if prediction[0] == 1:
        st.success(f"✅ Customer WILL Purchase  —  {proba:.1f}% confidence")
    else:
        st.error(f"❌ Customer will NOT Purchase  —  {100-proba:.1f}% confidence")

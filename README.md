# -Customer-Purchase-Prediction-
This project predicts whether a customer will make a purchase based on demographic and spending data.
It is built using Python, Logistic Regression, and Streamlit for deployment.

✨ Project Highlights:

📊 Exploratory Data Analysis (EDA)

📈 Data Visualization

🛠️ Feature Engineering

🤖 Machine Learning Model (Logistic Regression)

🌐 Deployment via Streamlit

📂 Dataset

The dataset simulates customer behavior:

Column Name	Description
Customer_ID	🆔 Unique customer identifier
Age	🎂 Customer age
Gender	👤 Male / Female
Monthly_Income	💰 Monthly income of the customer (₹)
Spending_Score	📊 Score representing spending behavior (0–100)
Purchased	✅ Target variable (1 = Purchased, 0 = Not Purchased)

Download dataset: customer_behavior_data.csv

✨ Key Features

🎂 Age

👤 Gender

💰 Monthly Income

📊 Spending Score

🎯 Target: Purchased

⚙️ How it Works

🔍 EDA & Visualization – Understand customer distribution and trends

🧹 Data Preprocessing – Encode categorical variables & scale features

🤖 Model Training – Logistic Regression to predict purchase likelihood

📊 Model Evaluation – Accuracy, Confusion Matrix, Classification Report

🌐 Deployment – Streamlit app for interactive predictions

🚀 Streamlit App

Run the app locally:

pip install -r requirements.txt
streamlit run app.py


Features of the App:

🖊️ Input customer details: Age, Gender, Income, Spending Score

🔮 Predict purchase likelihood

📊 Interactive & beginner-friendly UI

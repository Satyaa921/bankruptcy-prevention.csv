import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# App title and description
st.set_page_config(page_title="Bankruptcy Prediction", layout="centered")
st.title("Bankruptcy Risk Prediction App")
st.markdown("This app uses **logistic regression** to predict the risk of bankruptcy based on company financial indicators.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Copy of bankruptcy-prevention.csv", sep=";")
    df.columns = df.columns.str.strip()
    df["class"] = LabelEncoder().fit_transform(df["class"])  # 'Yes' → 1, 'No' → 0
    return df

data = load_data()

# Split data
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

st.subheader(" Model Evaluation Metrics")
col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
col1.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
col1.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
col2.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")
col2.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.2f}")

st.markdown("---")

# User Input Section
st.subheader("Predict Bankruptcy Risk")

def get_user_input():
    industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5)
    management_risk = st.slider("Management Risk", 0.0, 1.0, 0.5)
    financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.5)
    credibility = st.slider("Credibility", 0.0, 1.0, 0.5)
    competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5)
    operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5)

    input_data = pd.DataFrame([[
        industrial_risk,
        management_risk,
        financial_flexibility,
        credibility,
        competitiveness,
        operating_risk
    ]], columns=X.columns)

    return input_data

input_df = get_user_input()

# Predict
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Show Result
st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"The company is at HIGH RISK of Bankruptcy.\n\n**Probability:** {probability:.2f}")
else:
    st.success(f"The company is at LOW RISK of Bankruptcy.\n\n**Probability:** {probability:.2f}")

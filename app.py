import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# App title
st.title("Bankruptcy Prediction App")
st.write("This app predicts the risk of bankruptcy based on financial indicators.")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("bankruptcy-prevention.csv", sep=";")
    df.columns = df.columns.str.strip()
    df['class'] = LabelEncoder().fit_transform(df['class'])  # Yes: 1, No: 0
    return df

data = load_data()

X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

st.subheader(" Model Evaluation Metrics")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")
st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.2f}")
st.write(f"**ROC AUC:** {roc_auc_score(y_test, y_proba):.2f}")

# User input for prediction
st.subheader("🔍 Predict Bankruptcy Risk")

def get_user_input():
    industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5)
    management_risk = st.slider("Management Risk", 0.0, 1.0, 0.5)
    financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.5)
    credibility = st.slider("Credibility", 0.0, 1.0, 0.5)
    competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5)
    operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5)

    return pd.DataFrame([[
        industrial_risk,
        management_risk,
        financial_flexibility,
        credibility,
        competitiveness,
        operating_risk
    ]], columns=X.columns)

input_df = get_user_input()

# Make prediction
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Output result
st.subheader("📈 Prediction Result")
if prediction == 1:
    st.error(f"⚠High Risk of Bankruptcy\n\nProbability: {probability:.2f}")
else:
    st.success(f"Low Risk of Bankruptcy\n\nProbability: {probability:.2f}")

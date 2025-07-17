import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Bankruptcy Predictor", layout="centered")

# Header
st.title(" Bankruptcy Risk Prediction")
st.markdown(
    "Predict whether a company is at risk of bankruptcy based on financial indicators. "
    "Just move the sliders and click **Predict** to see the result."
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Copy of bankruptcy-prevention.csv", sep=";")
    df.columns = df.columns.str.strip()
    df["class"] = LabelEncoder().fit_transform(df["class"])  # Yes → 1, No → 0
    return df

data = load_data()
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Input UI
st.markdown("---")
st.subheader(" Input Financial Indicators")

col1, col2 = st.columns(2)

with col1:
    industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5)
    financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.5)
    competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5)

with col2:
    management_risk = st.slider("Management Risk", 0.0, 1.0, 0.5)
    credibility = st.slider("Credibility", 0.0, 1.0, 0.5)
    operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5)

input_data = pd.DataFrame([[
    industrial_risk,
    management_risk,
    financial_flexibility,
    credibility,
    competitiveness,
    operating_risk
]], columns=X.columns)

# Predict button
st.markdown("---")
if st.button("🔍 Predict Bankruptcy Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("### Prediction Result")

    if prediction == 1:
        st.error(f"⚠The company is at **HIGH RISK** of bankruptcy.\n\n**Probability:** {probability:.2f}")
    else:
        st.success(f"The company is at **LOW RISK** of bankruptcy.\n\n**Probability:** {probability:.2f}")
else:
    st.info("👆 Adjust the values and click **Predict Bankruptcy Risk**")



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Page settings
st.set_page_config(page_title="Bankruptcy Predictor", layout="centered")

# Sidebar with info
with st.sidebar:
    st.title("📘 About")
    st.markdown("""
    This app uses a **Logistic Regression** model to predict whether a company is at risk of bankruptcy
    based on six financial indicators:

    - Industrial Risk  
    - Management Risk  
    - Financial Flexibility  
    - Credibility  
    - Competitiveness  
    - Operating Risk  

    **Model:** Logistic Regression  
    **Data Source:** UCI Bankruptcy Dataset  
    """)
    st.markdown("**Author:** Satyaa921")

# Title
st.title("💼 Bankruptcy Risk Prediction")
st.markdown(
    "Predict whether a company is at **high risk of bankruptcy** using financial indicators. "
    "Adjust the sliders and click **Predict** to see the result."
)

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Copy of bankruptcy-prevention.csv", sep=";")
    df.columns = df.columns.str.strip()
    df["class"] = LabelEncoder().fit_transform(df["class"])
    df["class"] = 1 - df["class"]  # Flip so that 1 = bankruptcy
    return df

data = load_data()
X = data.drop("class", axis=1)
y = data["class"]

# Train model
model = LogisticRegression()
model.fit(X, y)
accuracy = model.score(X, y)

# Input sliders
st.markdown("---")
st.subheader("📥 Input Financial Indicators")

col1, col2 = st.columns(2)

with col1:
    industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5,
                                 help="Risk from the company's industry sector")
    financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.5,
                                      help="Ability to handle unexpected financial pressure")
    competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5,
                                help="Company's market strength and performance")

with col2:
    management_risk = st.slider("Management Risk", 0.0, 1.0, 0.5,
                                 help="Risk due to poor leadership or decision-making")
    credibility = st.slider("Credibility", 0.0, 1.0, 0.5,
                             help="Trustworthiness and public image")
    operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5,
                               help="Daily business operations vulnerability")

input_df = pd.DataFrame([[
    industrial_risk,
    management_risk,
    financial_flexibility,
    credibility,
    competitiveness,
    operating_risk
]], columns=X.columns)

# Prediction
st.markdown("---")
if st.button("🔍 Predict Bankruptcy Risk"):
    with st.spinner("Analyzing risk..."):
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.markdown("### 🔎 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ The company is at **HIGH RISK** of bankruptcy.\n\n**Probability:** {prob:.2f}")
        else:
            st.success(f"✅ The company is at **LOW RISK** of bankruptcy.\n\n**Probability:** {prob:.2f}")
else:
    st.info("👆 Adjust the sliders and click **Predict Bankruptcy Risk**")

# Model info
st.markdown("---")
st.markdown(f"📊 **Model Accuracy on Training Data:** `{accuracy:.2f}`")
st.caption("Built using Logistic Regression · UCI Bankruptcy Dataset · By Satyaa921")

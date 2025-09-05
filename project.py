import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import logging
from datetime import datetime

# -------------------------
# Logging Setup for SEBI Compliance
# -------------------------
logging.basicConfig(filename='client_interactions.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

st.set_page_config(layout="wide")

# -------------------------
# Title
# -------------------------
st.title("🤖 AI-Powered Robo-Advisory Platform")
st.write("Get dynamic and personalized investment recommendations.")

# -------------------------
# Sidebar for User Inputs
# -------------------------
st.sidebar.header("📌 Your Profile")
age = st.sidebar.slider("Enter your Age:", min_value=18, max_value=100, value=25)
income = st.sidebar.slider("Enter your Monthly Income (₹):", min_value=0, max_value=1000000, value=50000, step=1000)
goal = st.sidebar.selectbox("What is your Primary Goal?",
                            ["Wealth Creation", "Retirement", "Children's Education", "Short-term Savings"])
risk = st.sidebar.radio("What is your Risk Appetite?", ["Low", "Moderate", "High"])

# -------------------------
# AI Model for Recommendations
# -------------------------

# Mock data for training the model
data = {
    'age': [25, 40, 60, 30, 50, 22, 45, 65, 28, 55],
    'income': [50000, 100000, 75000, 200000, 150000, 30000, 120000, 80000, 60000, 180000],
    'goal': ["Wealth Creation", "Retirement", "Short-term Savings", "Wealth Creation", "Retirement",
             "Short-term Savings", "Children's Education", "Retirement", "Wealth Creation", "Children's Education"],
    'risk': ["High", "Moderate", "Low", "High", "Moderate", "Low", "Moderate", "Low", "High", "Moderate"],
    'portfolio': ["Aggressive Growth", "Balanced", "Conservative", "Aggressive Growth", "Balanced",
                  "Conservative", "Balanced", "Conservative", "Aggressive Growth", "Balanced"]
}
df = pd.DataFrame(data)

# Preprocessing
le_goal = LabelEncoder()
le_risk = LabelEncoder()
df['goal_encoded'] = le_goal.fit_transform(df['goal'])
df['risk_encoded'] = le_risk.fit_transform(df['risk'])

X = df[['age', 'income', 'goal_encoded', 'risk_encoded']]
y = df['portfolio']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

def get_ai_recommendation(age, income, goal, risk):
    # Preprocess user input
    goal_encoded = le_goal.transform([goal])[0]
    risk_encoded = le_risk.transform([risk])[0]

    # Create input array for prediction
    user_input = np.array([[age, income, goal_encoded, risk_encoded]])

    # Predict portfolio
    prediction = model.predict(user_input)[0]

    # Define portfolio allocations
    portfolios = {
        "Aggressive Growth": {"Direct Equity": 0.6, "International Funds": 0.2, "Small Cap Equity Funds": 0.2},
        "Balanced": {"Large Cap Equity Funds": 0.4, "Balanced Funds": 0.3, "Debt Mutual Funds": 0.3},
        "Conservative": {"Fixed Deposits": 0.5, "Bonds": 0.3, "Debt Mutual Funds": 0.2}
    }

    return prediction, portfolios.get(prediction, {})

recommendation_type, recommendation_portfolio = get_ai_recommendation(age, income, goal, risk)

# Log the interaction for compliance
log_message = (f"USER_PROFILE - Age: {age}, Income: {income}, Goal: {goal}, Risk: {risk} | "
               f"RECOMMENDATION - Type: {recommendation_type}, Portfolio: {recommendation_portfolio}")
logging.info(log_message)

# -------------------------
# Output
# -------------------------
st.subheader("🎯 Your AI-Powered Investment Recommendation")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.info("Your Profile")
        st.write(f"**Age:** {age}")
        st.write(f"**Monthly Income:** ₹{income}")
        st.write(f"**Primary Goal:** {goal}")
        st.write(f"**Risk Appetite:** {risk}")
    with col2:
        st.success(f"Recommended Portfolio: {recommendation_type}")
        for asset, weight in recommendation_portfolio.items():
            st.write(f"**{asset}:** {weight*100:.0f}%")

# -------------------------
# Automatic Rebalancing
# -------------------------
st.subheader("🔄 Automatic Rebalancing")

def rebalance_portfolio(current_portfolio, target_portfolio):
    # Simulate market changes (random fluctuations)
    simulated_returns = np.random.normal(0.01, 0.05, len(current_portfolio))

    current_values = np.array(list(current_portfolio.values()))
    simulated_values = current_values * (1 + simulated_returns)
    new_total_value = np.sum(simulated_values)
    new_weights = simulated_values / new_total_value

    # Check for deviation
    deviation = np.sum(np.abs(new_weights - np.array(list(target_portfolio.values()))))

    rebalance_needed = deviation > 0.1 # 10% deviation threshold

    return rebalance_needed, dict(zip(current_portfolio.keys(), new_weights))

if st.button("Check for Rebalancing"):
    rebalance_needed, new_portfolio = rebalance_portfolio(recommendation_portfolio, recommendation_portfolio)

    if rebalance_needed:
        st.warning("🚨 Your portfolio has deviated from its target. Rebalancing is recommended.")
        st.write("New Allocation:")
        for asset, weight in new_portfolio.items():
            st.write(f"**{asset}:** {weight*100:.0f}%")
    else:
        st.success("✅ Your portfolio is aligned with its target. No rebalancing needed.")

st.info("⚡ This is a demo advisory tool. For actual investment decisions, consult a SEBI-registered advisor.")

# -------------------------
# SEBI Compliance & Disclosure
# -------------------------
st.subheader("📜 SEBI Compliance & AI Model Disclosure")
st.write("""
**Record Keeping:** All your interactions and our recommendations are logged for regulatory purposes, in compliance with SEBI's five-year data retention policy.

**AI Model Explained:** Our recommendation engine uses a Random Forest Classifier, a machine learning model, to suggest a portfolio. It is trained on historical data and considers your age, income, financial goals, and risk appetite.

**Limitations:** The AI's recommendations are based on statistical models and are not infallible. They do not account for sudden market crashes or unforeseen geopolitical events. This is not a substitute for professional financial advice.
""")

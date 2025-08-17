import streamlit as st

# -------------------------
# Title
# -------------------------
st.title("🤖 Robo-Advisory Platform")
st.write("Get personalized investment recommendations based on your profile.")

# -------------------------
# User Inputs
# -------------------------
st.header("📌 Tell us about yourself")

age = st.number_input("Enter your Age:", min_value=18, max_value=100, value=25)
income = st.number_input("Enter your Monthly Income (₹):", min_value=0, value=50000)
goal = st.selectbox("What is your Primary Goal?", 
                    ["Wealth Creation", "Retirement", "Children's Education", "Short-term Savings"])
risk = st.radio("What is your Risk Appetite?", ["Low", "Moderate", "High"])

# -------------------------
# Recommendation Logic
# -------------------------
def get_recommendation(age, income, goal, risk):
    if risk == "Low":
        return "Debt Mutual Funds, Fixed Deposits, Bonds"
    elif risk == "Moderate":
        return "Balanced Funds, Index Funds, Large Cap Equity Funds"
    elif risk == "High":
        return "Small Cap Equity Funds, International Funds, Direct Equity"
    return "Diversified Portfolio"

recommendation = get_recommendation(age, income, goal, risk)

# -------------------------
# Output
# -------------------------
st.subheader("🎯 Your Investment Recommendation")
st.success(f"Based on your profile, we suggest: *{recommendation}*")

# Extra info
st.info("⚡ This is a demo advisory tool. For actual investment decisions, consult a SEBI-registered advisor.")
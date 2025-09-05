import streamlit as st

st.set_page_config(layout="wide")

# -------------------------
# Title
# -------------------------
st.title("🤖 Robo-Advisory Platform")
st.write("Get personalized investment recommendations based on your profile.")

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

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.info("Your Profile")
        st.write(f"**Age:** {age}")
        st.write(f"**Monthly Income:** ₹{income}")
        st.write(f"**Primary Goal:** {goal}")
        st.write(f"**Risk Appetite:** {risk}")
    with col2:
        st.success("Our Recommendation")
        st.write(recommendation)

st.info("⚡ This is a demo advisory tool. For actual investment decisions, consult a SEBI-registered advisor.")

# STEP 1: Set ngrok authtoken
from pyngrok import ngrok
import subprocess
from PIL import Image
import streamlit as st
import time


NGROK_AUTH_TOKEN = "enter"  # Replace with your token
ngrok.set_auth_token(NGROK_AUTH_TOKEN)


app_code = """
import streamlit as st
import pandas as pd
import os
import subprocess
import plotly.express as px
from PIL import Image
import time

from predict_risk import predict_risk
from prepare_feature_matrix import prepare_feature_matrix
from env import AssetAllocationEnv
from ppo import PPOAgent

# Set page config
st.set_page_config(page_title="AI Asset Allocator", layout="wide")
st.title("🛡️ AI-Powered Asset Allocation Advisor")

# 1. Basic Info
st.header("👤 Personal Information")
age = st.slider("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
education = st.selectbox("Education Level", options=["High School", "Bachelor's", "Master's", "PhD", "Other"])
marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced", "Widowed"])
marital_change = st.selectbox("Marital Status Change (past year)", options=[0, 1, 2, 3])

# 2. Financial Profile
st.header("💼 Financial Profile")
income = st.number_input("Annual Income ($)", min_value=0, step=1000)

st.markdown("### 💳 Credit Score")
credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)

loan_amount = st.number_input("Loan Amount ($)", min_value=0, step=500)
loan_purpose = st.selectbox("Loan Purpose", options=["Personal", "Auto", "Home", "Education", "Business"])
employment_status = st.selectbox("Employment Status", options=["Employed", "Unemployed", "Self-employed", "Retired", "Student"])
years_at_job = st.slider("Years at Current Job", min_value=0, max_value=50, value=5)
payment_history = st.selectbox("Payment History", options=["Good", "Fair", "Poor"])

st.markdown("### 📊 Debt-to-income ratio")
monthly_debt = st.number_input("Monthly Debt Payments ($)", min_value=0)
monthly_income = st.number_input("Monthly Gross Income ($)", min_value=1)
dti = round(monthly_debt / monthly_income, 2)
st.success(f"Calculated DTI: {dti:.2f}")

assets_value = st.number_input("Total Asset Value ($)", min_value=0, step=1000)

# 3. Background
st.header("🌍 Background")
dependents = st.slider("Number of Dependents", min_value=0, max_value=10, value=0)
city = st.text_input("City")
state = st.text_input("State")
country = st.text_input("Country")
prev_defaults = st.slider("Previous Loan Defaults", min_value=0, max_value=5, value=0)

# 4. Investment Profile
st.header("📈 Investment Preferences")
investment_horizon = st.selectbox("Select your investment horizon:", [
    "1 Day", "3 Days", "5 Days", "1 Week", "10 Days", "2 Weeks",
    "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "3 Years"
])


submit = st.button("✅ Submit and Predict Risk Rating")
if submit:
    # Save input to CSV
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Marital Status": marital_status,
        "Income": income,
        "Credit Score": credit_score,
        "Loan Amount": loan_amount,
        "Loan Purpose": loan_purpose,
        "Employment Status": employment_status,
        "Years at Current Job": years_at_job,
        "Payment History": payment_history,
        "Debt-to-Income Ratio": dti,
        "Assets Value": assets_value,
        "Number of Dependents": dependents,
        "City": city,
        "State": state,
        "Country": country,
        "Previous Defaults": prev_defaults,
        "Marital Status Change": marital_change,
        "Investment Horizon": investment_horizon
    }])

    os.makedirs("inputs", exist_ok=True)
    input_df.to_csv("inputs/risk_input.csv", index=False)
    st.info("✅ Saved input to `inputs/risk_input.csv`")

    # Predict risk (optional)
    try:
        rating = predict_risk()
        st.success(f"🧮 Predicted Risk Rating: **{rating}**")
    except Exception as e:
        st.warning(f"⚠️ Skipped risk prediction. Model not available or errored. Error: {e}")

# PPO RL Model Integration
if st.button("📊 Generate Portfolio Allocation"):
    with st.spinner("Running reinforcement learning model..."):
        result = subprocess.run(["python3", "run_rl.py"], capture_output=True, text=True)

        if result.returncode == 0:
            st.success("✅ Portfolio allocation completed!")

            df = pd.read_csv("rl_results.csv")
            df["Asset"] = df["Asset"].str.replace("_price_change", "", regex=False)

            def classify_asset(asset):
                if asset in [
                    "10-Year US Treasury", "5-Year US Treasury", "30-Year US Treasury",
                    "Corporate Bonds (LQD)", "High-Yield Bonds (HYG)", "Municipal Bonds (MUB)"
                ]:
                    return "Bonds"
                else:
                    return "Precious Metals"

            df["Category"] = df["Asset"].apply(classify_asset)

            fig = px.sunburst(
                df,
                path=["Category", "Asset"],
                values="Allocation (%)",
                title="Recommended Portfolio Allocation",
                color="Category",
                color_discrete_map={
                    "Bonds": "#1f77b4",
                    "Precious Metals": "#ff7f0e"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Portfolio Allocation (%)")
            st.dataframe(df)

        else:
            st.error("❌ RL model failed.")
            st.text(result.stderr)

"""


# Write app.py
with open("app.py", "w") as f:
    f.write(app_code)

# STEP 3: Launch Streamlit App
os.makedirs("inputs", exist_ok=True)
ngrok.kill()
subprocess.Popen(["streamlit", "run", "app.py"])
time.sleep(3)
public_url = ngrok.connect(8501)
print(f"🌐 Your app is live at: {public_url}")


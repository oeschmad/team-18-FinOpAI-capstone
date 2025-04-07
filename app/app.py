# STEP 1: Set ngrok authtoken
from pyngrok import ngrok

NGROK_AUTH_TOKEN = "2vOaH9lHauC1kgBndPtipPzh5DP_7ETtAUwfWeB9FkCFVgwMC"  # 🔁 Replace with your token
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

import streamlit as st
import pandas as pd
import os
# from bond_model import predict_bonds
import inspect
from predict_bonds import predict_bonds
print(inspect.signature(predict_bonds))
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Risk Form", layout="centered")
st.title("🛡️ Risk Assessment Form")

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

# Credit Score
st.markdown("### 💳 Credit Score")
auto_score = st.toggle("Estimate my credit score", value=True)
if auto_score:
    credit_category = st.selectbox("Select Credit Score Category", options=["Excellent", "Good", "Fair", "Poor", "Very Poor"])
    score_map = {
        "Excellent": 800,
        "Good": 720,
        "Fair": 650,
        "Poor": 580,
        "Very Poor": 500
    }
    credit_score = score_map[credit_category]
    st.info(f"Estimated Score: {credit_score}")
else:
    credit_score = st.slider("Enter Your Credit Score", min_value=300, max_value=850, value=680)

loan_amount = st.number_input("Loan Amount ($)", min_value=0, step=500)
loan_purpose = st.selectbox("Loan Purpose", options=["Personal", "Auto", "Home", "Education", "Business"])
employment_status = st.selectbox("Employment Status", options=["Employed", "Unemployed", "Self-employed", "Retired", "Student"])
years_at_job = st.slider("Years at Current Job", min_value=0, max_value=50, value=5)
payment_history = st.selectbox("Payment History", options=["Good", "Fair", "Poor"])

# Debt-to-Income Ratio
st.markdown("### 📊 Debt-to-income ratio")
auto_dti = st.toggle("Auto-calculate Debt-to-Income Ratio", value=True)
if auto_dti:
    monthly_debt = st.number_input("Monthly Debt Payments ($)", min_value=0)
    monthly_income = st.number_input("Monthly Gross Income ($)", min_value=1)
    dti = round(monthly_debt / monthly_income, 2)
    st.success(f"Calculated DTI: {dti:.2f}")
else:
    dti = st.number_input("Enter your Debt-to-Income Ratio manually (0 to 1)", min_value=0.0, max_value=1.0, step=0.01)

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

st.markdown("Investment Time Horizon : refers to how long you plan to keep your money invested before you need it?")

investment_horizon = st.selectbox("Select Your Investment Time Horizon", options=["1 Week", "1 Month", "3 Months", "6 Months", "More than 6 months"])

# 5. Submit
submit = st.button("✅ Submit and Predict Risk Rating")

if submit:
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

    st.success("✅ Input data prepared for risk model")
    st.dataframe(input_df)

    os.makedirs("inputs", exist_ok=True)
    input_df.to_csv("inputs/risk_input.csv", index=False)
    st.info("Saved input to `inputs/risk_input.csv`")

    try:
        from predict_risk import predict_risk
        rating = predict_risk()
        st.success(f"🧮 Predicted Risk Rating: **{rating}**")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Run bond model independently using selected horizon
if st.button("📉 Run Bond Model Based on Investment Horizon"):
    with st.spinner(f"Training model and predicting bond yields for: {investment_horizon}..."):
        try:
            forecast_df = predict_bonds(investment_horizon=investment_horizon)


            # Ensure Date is a column
            if forecast_df.index.name is not None:
                forecast_df = forecast_df.reset_index()
            if 'Date' not in forecast_df.columns:
                forecast_df.rename(columns={forecast_df.columns[0]: 'Date'}, inplace=True)

            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], errors='coerce')

            st.success("✅ Bond yield forecast completed!")
            st.subheader("📈 Forecasted Bond Yields")
            st.dataframe(forecast_df.head(5).style.format("{:.2f}"))

            # Plotting multiple yield curves
            st.subheader("📊 Yield Curves by Bond Type")
            yield_columns = [
                '10-Year US Treasury', '5-Year US Treasury', '30-Year US Treasury',
    'Corporate Bonds (LQD)', 'High-Yield Bonds (HYG)', 'Municipal Bonds (MUB)'
            ]
            melted_df = forecast_df.melt(id_vars='Date', value_vars=yield_columns,
                                          var_name='Bond Type', value_name='Yield')

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            for bond in melted_df['Bond Type'].unique():
                subset = melted_df[melted_df['Bond Type'] == bond]
                ax.plot(subset['Date'], subset['Yield'], label=bond)
            ax.set_xlabel("Date")
            ax.set_ylabel("Yield (%)")
            ax.set_title("Forecasted Bond Yield Curves")
            ax.legend(loc='upper left', fontsize='small')
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Failed to forecast bonds: {e}")


# STEP 3: Launch Streamlit App
import time
import os
os.makedirs("inputs", exist_ok=True)
ngrok.kill()
!streamlit run app.py &>/dev/null &
time.sleep(3)
public_url = ngrok.connect(8501)
print(f"🌐 Your app is live at: {public_url}")

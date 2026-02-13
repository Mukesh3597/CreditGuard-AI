# app/app.py
import os
import subprocess
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(page_title="CreditGuard AI", page_icon="🏦", layout="centered")

st.title("🏦 CreditGuard AI")
st.caption("Intelligent Loan Risk Assessment System (ML + SQL)")

# ---------------- Ensure model exists (Render/Cloud safe) ----------------
MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    try:
        # Train model on the server if missing
        subprocess.run(["python", "src/train.py"], check=True)
    except Exception as e:
        st.error(f"Auto-training failed on server: {e}")
        st.info("Tip: Check Render logs to see the exact train.py error.")
        st.stop()

# Load model
model = joblib.load(MODEL_PATH)

# ---------------- Import DB utils (optional but included) ----------------
# If you don't want DB, you can remove these lines and the history section.
from src.db_utils import init_db, insert_prediction, fetch_history

init_db()

# ---------------- UI ----------------
st.subheader("Enter Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=0, value=60000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=120000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

with col2:
    months_employed = st.number_input("Months Employed", min_value=0, max_value=500, value=48)
    num_credit_lines = st.number_input("Num Credit Lines", min_value=0, max_value=50, value=3)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
    loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=120, value=36)

with col3:
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=5.0, value=0.40)
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

has_mortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Home", "Education", "Business", "Other"])
has_cosigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])

st.divider()

# ---------------- Prediction ----------------
if st.button("Predict Risk"):
    sample = pd.DataFrame([{
        "Age": int(age),
        "Income": float(income),
        "LoanAmount": float(loan_amount),
        "CreditScore": int(credit_score),
        "MonthsEmployed": int(months_employed),
        "NumCreditLines": int(num_credit_lines),
        "InterestRate": float(interest_rate),
        "LoanTerm": int(loan_term),
        "DTIRatio": float(dti_ratio),
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }])

    pred = int(model.predict(sample)[0])
    proba = float(model.predict_proba(sample)[:, 1][0])

    st.subheader("Result")
    st.write(f"**Default Probability:** `{proba:.3f}`")

    # Probability bar (no custom colors)
    fig = plt.figure()
    plt.bar(["Default Probability"], [proba])
    plt.ylim(0, 1)
    st.pyplot(fig)

    if pred == 1:
        st.error("⚠️ High Risk: Likely to Default")
        st.write("Recommendation: Review manually before approval.")
    else:
        st.success("✅ Low Risk: Not Likely to Default")
        st.write("Recommendation: Eligible for approval (subject to policy checks).")

    # Save to SQLite
    row = {
        "Age": int(age),
        "Income": float(income),
        "LoanAmount": float(loan_amount),
        "CreditScore": int(credit_score),
        "MonthsEmployed": int(months_employed),
        "NumCreditLines": int(num_credit_lines),
        "InterestRate": float(interest_rate),
        "LoanTerm": int(loan_term),
        "DTIRatio": float(dti_ratio),

        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,

        "prediction": pred,
        "default_probability": proba
    }

    insert_prediction(row)
    st.info("✅ Saved to SQL Database (SQLite).")

# ---------------- History ----------------
st.divider()
st.subheader("📜 Prediction History (Last 20)")

history = fetch_history(limit=20)
st.dataframe(history, use_container_width=True)

# src/train.py  (FINAL - Render Friendly)

import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ✅ ensure root path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.preprocess import build_preprocessor

DATA_PATH = os.path.join(ROOT, "data", "raw", "Loan_default.csv")
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        # ✅ sample for Render (avoid heavy training)
        if len(df) > 30000:
            df = df.sample(30000, random_state=42).reset_index(drop=True)

        return df

    # ✅ fallback tiny demo dataset
    return pd.DataFrame([
        {"Age":30,"Income":60000,"LoanAmount":120000,"CreditScore":650,"MonthsEmployed":48,"NumCreditLines":3,"InterestRate":10,"LoanTerm":36,"DTIRatio":0.4,
         "Education":"Bachelor's","EmploymentType":"Full-time","MaritalStatus":"Married","HasMortgage":"Yes","HasDependents":"No","LoanPurpose":"Auto","HasCoSigner":"Yes","Default":0},
        {"Age":55,"Income":35000,"LoanAmount":180000,"CreditScore":420,"MonthsEmployed":6,"NumCreditLines":1,"InterestRate":22,"LoanTerm":60,"DTIRatio":0.8,
         "Education":"High School","EmploymentType":"Unemployed","MaritalStatus":"Single","HasMortgage":"No","HasDependents":"Yes","LoanPurpose":"Other","HasCoSigner":"No","Default":1},
        {"Age":41,"Income":90000,"LoanAmount":90000,"CreditScore":720,"MonthsEmployed":84,"NumCreditLines":4,"InterestRate":7,"LoanTerm":24,"DTIRatio":0.25,
         "Education":"Master's","EmploymentType":"Full-time","MaritalStatus":"Married","HasMortgage":"Yes","HasDependents":"Yes","LoanPurpose":"Home","HasCoSigner":"Yes","Default":0},
        {"Age":28,"Income":28000,"LoanAmount":150000,"CreditScore":390,"MonthsEmployed":12,"NumCreditLines":2,"InterestRate":19,"LoanTerm":48,"DTIRatio":0.7,
         "Education":"High School","EmploymentType":"Part-time","MaritalStatus":"Single","HasMortgage":"No","HasDependents":"No","LoanPurpose":"Business","HasCoSigner":"No","Default":1},
    ])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()

    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])

    if "Default" not in df.columns:
        raise ValueError("❌ Default column not found!")

    X = df.drop(columns=["Default"])
    y = df["Default"]

    preprocessor = build_preprocessor(X)

    rf = RandomForestClassifier(
        n_estimators=60,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("rf", rf)
    ])

    strat = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()

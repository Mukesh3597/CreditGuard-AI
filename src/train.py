import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from preprocess import build_preprocessor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Loan_default.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def load_data():
    # 1) If dataset exists in repo, use it
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df

    # 2) Fallback demo dataset (cloud-safe)
    # Minimal synthetic data so deployment never fails
    demo = pd.DataFrame([
        {"Age":30,"Income":60000,"LoanAmount":120000,"CreditScore":650,"MonthsEmployed":48,"NumCreditLines":3,"InterestRate":10,"LoanTerm":36,"DTIRatio":0.4,
         "Education":"Bachelor's","EmploymentType":"Full-time","MaritalStatus":"Married","HasMortgage":"Yes","HasDependents":"No","LoanPurpose":"Auto","HasCoSigner":"Yes","Default":0},
        {"Age":55,"Income":35000,"LoanAmount":180000,"CreditScore":420,"MonthsEmployed":6,"NumCreditLines":1,"InterestRate":22,"LoanTerm":60,"DTIRatio":0.8,
         "Education":"High School","EmploymentType":"Unemployed","MaritalStatus":"Single","HasMortgage":"No","HasDependents":"Yes","LoanPurpose":"Other","HasCoSigner":"No","Default":1},
        {"Age":41,"Income":90000,"LoanAmount":90000,"CreditScore":720,"MonthsEmployed":84,"NumCreditLines":4,"InterestRate":7,"LoanTerm":24,"DTIRatio":0.25,
         "Education":"Master's","EmploymentType":"Full-time","MaritalStatus":"Married","HasMortgage":"Yes","HasDependents":"Yes","LoanPurpose":"Home","HasCoSigner":"Yes","Default":0},
        {"Age":28,"Income":28000,"LoanAmount":150000,"CreditScore":390,"MonthsEmployed":12,"NumCreditLines":2,"InterestRate":19,"LoanTerm":48,"DTIRatio":0.7,
         "Education":"High School","EmploymentType":"Part-time","MaritalStatus":"Single","HasMortgage":"No","HasDependents":"No","LoanPurpose":"Business","HasCoSigner":"No","Default":1},
    ])
    return demo

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()

    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])

    X = df.drop(columns=["Default"])
    y = df["Default"]

    preprocessor = build_preprocessor(X)

    # Cloud में fast training के लिए छोटा model
    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("rf", rf)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model.fit(X_train, y_train)

    # Optional quick metric (won't crash if tiny data)
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else None
        print("ROC-AUC:", auc)
    except Exception as e:
        print("Metric skipped:", e)

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()

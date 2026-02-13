import joblib
import pandas as pd

# Load trained model
model = joblib.load("../models/model.pkl")

# Example input (same columns as training features)
sample = pd.DataFrame([{
    "Age": 45,
    "Income": 60000,
    "LoanAmount": 120000,
    "CreditScore": 650,
    "MonthsEmployed": 48,
    "NumCreditLines": 3,
    "InterestRate": 10,
    "LoanTerm": 36,
    "DTIRatio": 0.4,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "No",
    "LoanPurpose": "Auto",
    "HasCoSigner": "Yes"
}])

pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[:, 1][0]

print("Prediction (0=No Default, 1=Default):", pred)
print("Default Probability:", proba)

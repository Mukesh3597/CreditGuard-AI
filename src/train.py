import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from preprocess import build_preprocessor


# ---------- Paths ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Loan_default.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def main():
    # Ensure models folder exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Drop ID
    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])

    # Split X, y
    X = df.drop(columns=["Default"])
    y = df["Default"]

    # Preprocess + Model
    preprocessor = build_preprocessor(X)

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("rf", rf)
    ])

    # Train-test split with stratify (important for imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model trained and saved successfully: {MODEL_PATH}")


if __name__ == "__main__":
    main()

# 🏦 CreditGuard AI

CreditGuard AI is an end-to-end Loan Default Prediction System built using Machine Learning, Streamlit, and SQLite.  
It performs real-time credit risk assessment and stores prediction history in a database.

---

## 🚀 Live Demo

🔗 https://creditguard-ai.onrender.com

---

## 📌 Features

- ✅ Loan Default Prediction using RandomForest
- ✅ Handles Class Imbalance using class_weight="balanced"
- ✅ OneHot Encoding for categorical features
- ✅ Modular Project Structure (Clean Architecture)
- ✅ Streamlit Web Application
- ✅ SQLite Database Integration
- ✅ Prediction History Tracking
- ✅ Cloud Deployment (Render)

---

## 🧠 Machine Learning Details

- Model: RandomForestClassifier
- Preprocessing: ColumnTransformer + OneHotEncoder
- Train/Test Split: Stratified
- Imbalance Handling: class_weight = "balanced"
- Evaluation Metrics:
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

---

## 📂 Project Structure

CreditGuard-AI/
│
├── app/
│ └── app.py # Streamlit Web UI
│
├── src/
│ ├── train.py # Model Training
│ ├── preprocess.py # Data Preprocessing
│ ├── db_utils.py # Database Operations
│ └── predict.py # Testing Predictions
│
├── sql/
│ └── 01_create_tables.sql # Database Table Schema
│
├── models/ # Saved Model (local)
├── data/raw/ # Raw Dataset
├── requirements.txt
└── README.md

---

## 🗄️ Database

SQLite database stores:

- User input data
- Model prediction
- Default probability
- Timestamp

---

## ⚙️ Installation (Local Setup)

```bash
git clone https://github.com/Mukesh3597/CreditGuard-AI.git
cd CreditGuard-AI
pip install -r requirements.txt
Run locally:

python -m streamlit run app/app.py
🌐 Deployment

This project is deployed on Render Cloud
Free tier auto-sleeps after inactivity.

🎯 Use Case

This project simulates a real-world financial risk assessment system that can be used by:

Banks

FinTech companies

NBFCs

Lending startups

👨‍💻 Author

Mukesh Pratap
BCA (Hons) – Machine Learning Enthusiast
GitHub: https://github.com/Mukesh3597

⭐ Future Improvements

Admin dashboard analytics

Model comparison (XGBoost, Logistic Regression)

Authentication system

Docker support

REST API integration

📌 Disclaimer

This project is for educational and demonstration purposes only.


---

# 🎯 अब क्या करो?

1. GitHub में `README.md` open करो  
2. Edit  
3. ऊपर वाला पूरा content paste करो  
4. Commit changes  

---

अगर तुम चाहो तो मैं:

- 🔥 README को और elite बनाऊँ (badges + architecture diagram)
- 🎯 LinkedIn post का professional draft दूँ
- 📊 System architecture diagram बना दूँ

बताओ next क्या upgrade करें? 🚀

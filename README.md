# Telecom-Customer-Churn-Analysis-and-Prediction
This project aims to predict customer churn for a telecom company using machine learning models. The analysis involves exploratory data analysis (EDA) to identify key churn factors, feature engineering, and applying various classification algorithms.
Class imbalance was handled using oversampling techniques like RandomOverSampler and SMOTE to improve recall for churners.
The goal is to help telecom providers proactively identify at-risk customers and implement targeted retention strategies.

📊 Dataset
The dataset contains telecom customer information, including:

Demographic details: Gender, SeniorCitizen, Partner, Dependents

Service details: PhoneService, InternetService, Contract, PaymentMethod

Usage & charges: Tenure, MonthlyCharges, TotalCharges

Target variable: Churn (Yes/No)

🔍 Steps Followed
Data Preprocessing

Handled missing values & data type corrections

Encoded categorical variables

Standardized numerical features

Exploratory Data Analysis (EDA)

Visualized churn rates by contract type, tenure, payment method

Correlation analysis between churn and other variables

Handling Class Imbalance

Applied RandomOverSampler & SMOTE

Compared oversampling vs. class weighting

Model Building

Logistic Regression

Random Forest

AdaBoost

XGBoost

Stacking Classifier

Hyperparameter Tuning

Used GridSearchCV & cross-validation

Optimized for F1-score (balance between recall & precision)

Model Evaluation

Accuracy, Precision, Recall, F1-score

Confusion matrix

ROC-AUC curve

⚙️ Technologies Used
Python

Pandas, NumPy – Data manipulation

Matplotlib, Seaborn – Visualization

Scikit-learn – Machine learning models

Imbalanced-learn – SMOTE & oversampling

XGBoost – Gradient boosting

📈 Results
Best Recall for Churn: 0.82 (Stacking Classifier + Oversampling)

Best Balanced Performance: Accuracy ~79%, Recall ~0.70, Precision ~0.60 (Threshold tuning)

High recall helps catch most churners, enabling better retention strategies.

🚀 How to Run
bash
Copy
Edit
# Clone repository
git clone https://github.com/your-username/telecom-churn-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run script
python churn_prediction.py

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 17:03:12 2025

@author: Lenovo
"""

import pandas as pd
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the Dataset
telecom_cust = pd.read_csv(r'Telco_Customer_Churn.csv')

# Data Preprocessing
telecom_cust['TotalCharges'] = pd.to_numeric(telecom_cust['TotalCharges'], errors = 'coerce')
telecom_cust['TotalCharges'].fillna(0, inplace=True)

# Convert Churn columns value to Binary Value
label_encoder = LabelEncoder()
# churn
telecom_cust['Churn'] = label_encoder.fit_transform(telecom_cust['Churn'])

telecom_cust['InternetService'] = label_encoder.fit_transform(telecom_cust['InternetService'])
telecom_cust['Contract'] = label_encoder.fit_transform(telecom_cust['Contract'])

# Select the limited features
selected_features = ['tenure', 'InternetService', 'Contract', 'MonthlyCharges', 'TotalCharges']

# Identify Independent and dependent variable
X = telecom_cust[selected_features]
y = telecom_cust['Churn']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=101)
model.fit(X, y)

# Save the trained model
dump(model, 'random_forest_model.joblib')











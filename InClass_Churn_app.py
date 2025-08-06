# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 20:02:20 2025

@author: Lenovo
"""

import streamlit as st
from joblib import load

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the model
model = load('random_forest_model.joblib')

# Build the UI using Streamlit
st.title("Customer Churn Prediction Application")

# Input field in UI
st.header("Please Enter Customer Information")

tenure = st.number_input("Tenure (in month)", min_value = 0, max_value = 70, value = 1)

internet_service = st.selectbox("Internet Service", ("DSL", "Fiber Optic", 'No'))

contract = st.selectbox("Contract", ("Month-to-Month", "One-Year", "Two-Year"))

monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=200, value = 50)

total_charges = st.number_input("Total Charges", min_value=200, max_value=5000, value = 250)

# Map input values to numeric using the label mapping
label_mapping = {
    "DSL" : 0,
    "Fiber Optic" : 1,
    "No": 2,
    "Month-to-Month" : 0,
    "One-Year" : 1,
    "Two-Year" : 2}
internet_service = label_mapping[internet_service]
contract = label_mapping[contract]

# Make a prediction using the model
prediction = model.predict([[tenure, internet_service, contract, monthly_charges, total_charges]])

# Display the prediction in UI
st.header("Prediction Result")

if prediction[0] == 0:
    st.success("Customer will stay with us")
else:
    st.error("Customer will churn away")









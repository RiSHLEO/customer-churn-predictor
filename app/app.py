import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model, threshold, and feature columns
with open('../models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/threshold.pkl', 'rb') as f:
    threshold = pickle.load(f)

with open('../models/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# App title
st.title('Customer Churn Predictor')
st.write('Enter customer details below to predict whether they are likely to churn.')

# Input fields
st.subheader('Customer Details')

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider('Months as customer', 0, 72, 12)
    monthly_charges = st.number_input('Monthly Charges (£)', 0.0, 200.0, 50.0)
    contract = st.selectbox('Contract Type', 
                            ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox('Internet Service', 
                                    ['DSL', 'Fiber optic', 'No'])

with col2:
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Has Partner', ['No', 'Yes'])
    dependents = st.selectbox('Has Dependents', ['No', 'Yes'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])

# Predict button
if st.button('Predict Churn Risk'):

    # Build input dict with all zeros first
    input_dict = {col: 0 for col in feature_columns}

    # Fill in numeric values
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = monthly_charges * tenure
    input_dict['SeniorCitizen'] = 1 if senior_citizen == 'Yes' else 0
    input_dict['Partner'] = 1 if partner == 'Yes' else 0
    input_dict['Dependents'] = 1 if dependents == 'Yes' else 0
    input_dict['PaperlessBilling'] = 1 if paperless_billing == 'Yes' else 0

    # Fill in engineered features
    input_dict['AvgMonthlySpend'] = monthly_charges
    input_dict['TotalServices'] = 1
    input_dict['IsNewCustomer'] = 1 if tenure <= 12 else 0
    input_dict['HighValueCustomer'] = 1 if monthly_charges > 64.76 else 0

    # Fill in contract type
    if contract == 'One year':
        input_dict['Contract_One year'] = 1
    elif contract == 'Two year':
        input_dict['Contract_Two year'] = 1

    # Fill in internet service
    if internet_service == 'Fiber optic':
        input_dict['InternetService_Fiber optic'] = 1
    elif internet_service == 'No':
        input_dict['InternetService_No'] = 1

    # Convert to dataframe and predict
    input_df = pd.DataFrame([input_dict])
    probability = model.predict_proba(input_df)[0][1]
    prediction = 1 if probability >= threshold else 0

    # Show result
    st.subheader('Prediction Result')

    if prediction == 1:
        st.error(f'High Churn Risk — {probability:.0%} probability of churning')
        st.write('**Recommended actions:**')
        st.write('- Offer a discount to switch to an annual contract')
        st.write('- Assign a customer success manager')
        st.write('- Follow up with a satisfaction survey')
    else:
        st.success(f'Low Churn Risk — {probability:.0%} probability of churning')
        st.write('This customer is likely to stay.')
import joblib
import streamlit as st
import pandas as pd

# Load the model
cb_loaded = joblib.load('car_price_model.joblib')

# Define the user input form
st.write('# Car Price Prediction')
present_price = st.number_input('Present Price', min_value=0.0, max_value=100000.0)
present_price = present_price / 58823.53
kms_driven = st.number_input('Kms Driven', min_value=0, max_value=1000000, step=1000)
owner = st.selectbox('Owner', [0, 1, 2, 3])
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
age = st.number_input('Age', min_value=0, max_value=30, step=1)

input_list = [present_price, kms_driven, owner]

if fuel_type == 'Diesel':
    input_list.append(1)
else:
    input_list.append(0)

if fuel_type == 'Petrol':
    input_list.append(1)
else:
    input_list.append(0)

if seller_type == 'Individual':
    input_list.append(1)
else:
    input_list.append(0)

if transmission == 'Manual':
    input_list.append(1)
else:
    input_list.append(0)

input_list.append(age)

inputs_df = pd.DataFrame([input_list], columns=['Present_Price', 'Kms_Driven', 'Owner',
                                                'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
                                                'Seller_Type_Individual', 'Transmission_Manual',
                                                'Age'])
prediction = cb_loaded.predict(inputs_df)[0]
prediction_dollars = prediction * 58823.53

# Display the prediction to the user
st.header(f'Predicted selling price: {prediction_dollars:.2f} $')

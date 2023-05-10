import joblib
import streamlit as st
import pandas as pd

# Load the model
cb_loaded = joblib.load('car_price_model.joblib')

# Define the user input form
st.write('# Car Price Prediction')
year = st.slider('Year', 2003, 2023, 2014)
selling_price = st.number_input('Selling Price', min_value=0.0, max_value=50.0, step=0.1)
present_price = st.number_input('Present Price', min_value=0.0, max_value=50.0, step=0.1)
kms_driven = st.number_input('Kms Driven', min_value=0, max_value=1000000, step=1000)
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.selectbox('Owner', [0, 1, 2, 3])

# Make a prediction based on the user input
inputs = [[year, selling_price, present_price, kms_driven, fuel_type, seller_type, transmission, owner]]
inputs_df = pd.DataFrame(inputs, columns=['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
                                          'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])
inputs_df = pd.get_dummies(inputs_df, drop_first=True)
prediction = cb_loaded.predict(inputs_df)[0]

# Display the prediction to the user
st.write(f'Predicted selling price: {prediction:.2f} lakhs')
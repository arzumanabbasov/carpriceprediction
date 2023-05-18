import joblib
import streamlit as st
import pandas as pd

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('car_price_model.joblib')

# Define data preprocessing functions
def preprocess_input(present_price, kms_driven, owner, fuel_type, seller_type, transmission, age):
    input_list = [present_price / 58823.53, kms_driven, owner]
    
    input_list.append(1 if fuel_type == 'Diesel' else 0)
    input_list.append(1 if fuel_type == 'Petrol' else 0)
    
    input_list.append(1 if seller_type == 'Individual' else 0)
    
    input_list.append(1 if transmission == 'Manual' else 0)
    
    input_list.append(age)
    
    inputs_df = pd.DataFrame([input_list], columns=['Present_Price', 'Kms_Driven', 'Owner',
                                                    'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
                                                    'Seller_Type_Individual', 'Transmission_Manual',
                                                    'Age'])
    return inputs_df

# Load the model outside of the main app function
cb_loaded = load_model()

# Define the user input form
def user_input_form():
    st.write('# Car Price Prediction')
    present_price = st.number_input('Present Price', min_value=0, max_value=100000)
    kms_driven = st.number_input('Kms Driven', min_value=0, max_value=1000000, step=1000)
    owner = st.selectbox('Owner', [0, 1, 2, 3])
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    age = st.number_input('Age', min_value=0, max_value=30, step=1)
    return present_price, kms_driven, owner, fuel_type, seller_type, transmission, age

# Run the app
def main():
    # Render the user input form
    present_price, kms_driven, owner, fuel_type, seller_type, transmission, age = user_input_form()
    
    # Preprocess the input data
    inputs_df = preprocess_input(present_price, kms_driven, owner, fuel_type, seller_type, transmission, age)

    # Perform the prediction
    prediction = cb_loaded.predict(inputs_df)[0]
    prediction_dollars = prediction * 58823.53

    # Display the prediction to the user
    st.header(f'Predicted selling price: {prediction_dollars:.2f} $')

if __name__ == '__main__':
    main()

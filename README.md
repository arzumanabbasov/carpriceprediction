# Car Price Prediction App
This is a simple web app that predicts the selling price of a car based on various features such as its age, present price, kilometers driven, fuel type, seller type, transmission, and number of owners. The app is built using Streamlit and a machine learning model trained on a dataset of car listings.

## Getting Started
To use this app, you need to have Python 3 installed on your computer. You can then clone this repository and install the required Python packages using the following command:`pip install -r requirements.txt`

Once the packages are installed, you can run the app using the following command:`streamlit run app.py`
This will launch the app in your default web browser.

## Usage
To use the app, simply fill in the input fields with the relevant information about the car you want to sell. The app will then predict the selling price of the car based on the machine learning model trained on the dataset.

## Model Details
The machine learning model used in this app is a CatBoost regressor trained on a dataset of car listings. The model takes in various features such as age, present price, kilometers driven, fuel type, seller type, transmission, and number of owners, and predicts the selling price of the car. The model has an R-squared score of 0.96 on test data, indicating a good fit.

## Acknowledgments
The dataset used in this app is sourced from the Kaggle dataset repository: https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho. The CatBoost algorithm is developed by Yandex and is available under the Apache License 2.0.

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model
with open('LinearRegressionModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the car data
car = pd.read_csv('cleaned_car.csv')

# Streamlit application
st.title("Car Price Prediction")

# Sidebar for input features
st.sidebar.header("Input Features")

# Select Company
companies = sorted(car['company'].unique())
companies.insert(0, 'Select Company')
selected_company = st.sidebar.selectbox('Company', companies)

# Select Car Model
car_models = sorted(car['name'].unique())
selected_car_model = st.sidebar.selectbox('Car Model', car_models)

# Select Year
years = sorted(car['year'].unique(), reverse=True)
selected_year = st.sidebar.selectbox('Year', years)

# Select Fuel Type
fuel_types = car['fuel_type'].unique()
selected_fuel_type = st.sidebar.selectbox('Fuel Type', fuel_types)

# Input Kms Driven
kms_driven = st.sidebar.number_input('Kms Driven', min_value=0)

# Predict button
if st.sidebar.button('Predict'):
    if selected_company != 'Select Company':
        # Prepare input data
        input_data = pd.DataFrame(
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
            data=np.array([selected_car_model, selected_company, selected_year, kms_driven, selected_fuel_type]).reshape(1, 5)
        )
        
        # Make prediction
        prediction = model.predict(input_data)
        st.write(f'The predicted price is: ${np.round(prediction[0], 2)}')
    else:
        st.error('Please select a company')


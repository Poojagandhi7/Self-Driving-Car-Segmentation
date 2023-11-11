import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained Random Forest Regressor model for insurance cost prediction
with open("medical_insurance_cost.pkl", "rb") as model_file:
    random_forest_regressor = pickle.load(model_file)

# Load the trained model for life expectancy prediction
model_path = "Life_Expectancy_Prediction.pkl"
with open(model_path, "rb") as model_file:
    life_expectancy_model = pickle.load(model_file)

# Encoding mappings for insurance cost prediction
sex_mapping = {'Male': 0, 'Female': 1}
smoker_mapping = {'Yes': 0, 'No': 1}
region_mapping = {'Southeast': 0, 'Southwest': 1, 'Northeast': 2, 'Northwest': 3}

# Function to predict life expectancy
def predict_life_expectancy(features):
    return life_expectancy_model.predict(features)

# Page title and subtitle for insurance cost prediction
st.title('Healthcare Services')
st.subheader('Insurance Cost Prediction')

# Input fields for insurance cost prediction
age = st.number_input('Age', min_value=18, max_value=100, step=1)
sex = st.radio('Sex', ['Male', 'Female'])
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, step=0.1)
children = st.number_input('Number of Children', min_value=0, max_value=10, step=1)
smoker = st.radio('Smoker', ['Yes', 'No'])
region = st.radio('Region', ['Southeast', 'Southwest', 'Northeast', 'Northwest'])

sex = sex_mapping[sex]
smoker = smoker_mapping[smoker]
region = region_mapping[region]

# Button to trigger the insurance cost prediction
if st.button('Predict Insurance Cost'):
    input_data = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)
    prediction = random_forest_regressor.predict(input_data)
    st.subheader('Insurance Cost Prediction:')
    st.write('Predicted Insurance Cost:', prediction[0])

# Add a line to separate the sections
st.write('<hr>', unsafe_allow_html=True)

# Subtitle for life expectancy prediction
st.subheader('Life Expectancy Prediction')

# Input fields for life expectancy prediction
year = st.number_input("Enter the year:", min_value=2000, max_value=2025, value=2020)
status = st.radio("Select the country status:", ["Developed", "Developing"])
population = st.number_input("Enter the population:", min_value=0, value=20835722)
hepatitis_b = st.number_input("Enter hepatitis B coverage (%):", min_value=0, max_value=100, value=87)
measles = st.number_input("Enter measles cases:", min_value=0, value=2500)
alcohol = st.number_input("Enter alcohol consumption (liters):", min_value=0.0, value=7.0)
bmi = st.number_input("Enter BMI:", min_value=0.0, value=28.0)
polio = st.number_input("Enter polio coverage (%):", min_value=0, max_value=100, value=90)
diphtheria = st.number_input("Enter diphtheria coverage (%):", min_value=0, max_value=100, value=88)
hiv_aids = st.number_input("Enter HIV/AIDS prevalence (%):", min_value=0.0, value=2.0)
gdp = st.number_input("Enter GDP per capita:", min_value=0, value=51386)

status_mapping = {'Developed': 0, 'Developing': 1}
status = status_mapping[status]

# Button to trigger the life expectancy prediction
if st.button('Predict Life Expectancy'):
    user_inputs = pd.DataFrame({
        'year': [year],
        'status': [status],
        'population': [population],
        'hepatitis_b': [hepatitis_b],
        'measles': [measles],
        'alcohol': [alcohol],
        'bmi': [bmi],
        'polio': [polio],
        'diphtheria': [diphtheria],
        'hiv/aids': [hiv_aids],
        'gdp': [gdp],
    })
    prediction = predict_life_expectancy(user_inputs)
    st.subheader('Life Expectancy Prediction:')
    st.write('Predicted Life Expectancy:', prediction[0])

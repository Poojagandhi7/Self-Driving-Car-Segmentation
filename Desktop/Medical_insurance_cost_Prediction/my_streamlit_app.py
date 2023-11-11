import streamlit as st
import pickle
import numpy as np

# Load the trained Random Forest Regressor model
with open("medical_insurance_cost.pkl", "rb") as model_file:
    random_forest_regressor = pickle.load(model_file)

# Page title and header
st.title('Insurance Cost Prediction')

# Sidebar for user input
st.sidebar.header('User Input')

# Create input fields for user data
age = st.sidebar.number_input('Age', min_value=18, max_value=100, step=1)
sex = st.sidebar.radio('Sex', ['Male', 'Female'])
bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=60.0, step=0.1)
children = st.sidebar.number_input('Number of Children', min_value=0, max_value=10, step=1)
smoker = st.sidebar.radio('Smoker', ['Yes', 'No'])
region = st.sidebar.radio('Region', ['Southeast', 'Southwest', 'Northeast', 'Northwest'])

# Encoding mappings
sex_mapping = {'Male': 0, 'Female': 1}
smoker_mapping = {'Yes': 0, 'No': 1}
region_mapping = {'Southeast': 0, 'Southwest': 1, 'Northeast': 2, 'Northwest': 3}

sex = sex_mapping[sex]
smoker = smoker_mapping[smoker]
region = region_mapping[region]

# Create a button to trigger the prediction
if st.sidebar.button('Predict'):
    input_data = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)
    prediction = random_forest_regressor.predict(input_data)
    st.write('Predicted Insurance Cost:', prediction[0])

# You can run this script using the Streamlit command in Visual Studio Code:
# streamlit run your_script.py

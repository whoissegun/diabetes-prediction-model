import streamlit as st
from utils import calculate_dpf, predict_diabetes
from diabetes_predictive_model import DiabetesPredictionModelV1
import torch


def main():

    # Load the model
    model = torch.load("model.pth")
    model.eval()




    st.title('Diabetes Prediction Form')

    # User input fields
    pregnancies = st.number_input('Number of pregnancies', min_value=0, max_value=20)
    glucose = st.number_input('Glucose level', min_value=0, max_value=200)
    blood_pressure = st.number_input('Blood pressure', min_value=0, max_value=200)
    skin_thickness = st.number_input('Skin thickness', min_value=0, max_value=100)
    insulin = st.number_input('Insulin level', min_value=0, max_value=900)
    bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=100.0)
    age = st.number_input('Age', min_value=10, max_value=120)


    dpf = 0
    if st.button('Calculate Diabetes Pedigree Function (DPF)'):
        dpf = calculate_dpf(age, bmi, glucose, blood_pressure)
        st.write('Your calculated DPF is:', dpf)


    if st.button('Predict Diabetes'):
        
        prediction = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, model)
        if prediction == 1:
            st.success('There is a high likelihood of diabetes.')
        else:
            st.success('There is a low likelihood of diabetes.')


if __name__ == '__main__':
    main()
import streamlit as st
import numpy as np

import torch

class DiabetesPredictionModelV1(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(in_features=16, out_features=32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(in_features=32, out_features=16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(in_features=16, out_features=8),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(in_features=8, out_features=4),
            torch.nn.BatchNorm1d(4),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(in_features=4, out_features=out_features),
        )

    def forward(self, X):
        return self.layers(X).squeeze(dim=1)

# Assuming you have already saved your entire model and have it in the correct directory:
model = torch.load("model.pth")
model.eval()


# Define the prediction function using the loaded model
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    # Prepare the input data
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    features_tensor = torch.from_numpy(features).float()  # Convert to PyTorch tensor
    
    # Make a prediction
    with torch.no_grad():  # Disable gradient computation for inference
        prediction = model(features_tensor)
        predicted_class = torch.round(torch.sigmoid(prediction))  # Assuming binary classification with sigmoid output
    return predicted_class.item()  # Get the prediction result as a Python number


# Define a function to calculate a simplified DPF
def calculate_dpf(age, bmi, glucose, blood_pressure):
    # This is a very simplified formula and should be replaced with an actual calculation based on your research
    age_factor = age / 100
    bmi_factor = bmi / 50
    glucose_factor = glucose / 200
    bp_factor = blood_pressure / 120
    dpf = (age_factor + bmi_factor + glucose_factor + bp_factor) / 4
    return round(dpf, 2)

# Title of the web app
st.title('Diabetes Prediction Form')

# User input fields
pregnancies = st.number_input('Number of pregnancies', min_value=0, max_value=20)
glucose = st.number_input('Glucose level', min_value=0, max_value=200)
blood_pressure = st.number_input('Blood pressure', min_value=0, max_value=200)
skin_thickness = st.number_input('Skin thickness', min_value=0, max_value=100)
insulin = st.number_input('Insulin level', min_value=0, max_value=900)
bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=100.0)
age = st.number_input('Age', min_value=10, max_value=120)

# Calculate DPF button
dpf = 0
if st.button('Calculate Diabetes Pedigree Function (DPF)'):
    dpf = calculate_dpf(age, bmi, glucose, blood_pressure)
    st.write('Your calculated DPF is:', dpf)

# Predict button (assuming a prediction function called `predict_diabetes` exists)
if st.button('Predict Diabetes'):
    # Assuming prediction function takes all the features including calculated DPF
    prediction = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    if prediction == 1:
        st.success('There is a high likelihood of diabetes.')
    else:
        st.success('There is a low likelihood of diabetes.')

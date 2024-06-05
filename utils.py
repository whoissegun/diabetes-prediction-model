import torch
import numpy as np

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, model):
    

    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    features_tensor = torch.from_numpy(features).float()  # Convert to PyTorch tensor
    
    # Make a prediction
    with torch.no_grad():  # Disable gradient computation for inference
        prediction = model(features_tensor)
        predicted_class = torch.round(torch.sigmoid(prediction))  # Assuming binary classification with sigmoid output
    return predicted_class.item()  # Get the prediction result as a Python number


def calculate_dpf(age, bmi, glucose, blood_pressure):
    # This is a very simplified formula I got from chat GPT-4 not sure if it's accurate
    age_factor = age / 100
    bmi_factor = bmi / 50
    glucose_factor = glucose / 200
    bp_factor = blood_pressure / 120
    dpf = (age_factor + bmi_factor + glucose_factor + bp_factor) / 4
    return round(dpf, 2)

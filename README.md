# Diabetes Prediction App

This diabetes prediction app utilizes a machine learning model to predict the likelihood of diabetes based on several medical indicators. Built with PyTorch and Streamlit, the app offers an interactive user interface for entering medical data and receiving instant predictions.

The deployed model can be accessed at: https://divine-diabetes-prediction-model.streamlit.app/

## Features

- Predict diabetes likelihood based on medical inputs like glucose levels, blood pressure, and more.
- Calculate Diabetes Pedigree Function (DPF) dynamically based on user input.
- Interactive UI built with Streamlit.
- Model accuracy of 81%.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
git clone <https://github.com/whoissegun/diabetes-prediction-model.git>

2. Install required Python packages:
pip install -r requirements.txt



## Usage

To run the app locally:

1. Navigate to the project directory:
cd diabetes-prediction-model

2. Run the Streamlit app:
streamlit run streamlit_app.py




## Model Details

The model is a binary classification model developed with PyTorch. Here are the specifics of the model architecture:

- Input features: 8 (Number of pregnancies, Glucose, Blood pressure, Skin thickness, Insulin, BMI, DPF, Age)
- Output: Binary (1 = high likelihood of diabetes, 0 = low likelihood)
- Layers: Multiple linear layers with batch normalization, ReLU activations, and dropout.
- Optimizer: Adam
- Loss Function: BCEWithLogitsLoss

## Data

The dataset used for training is sourced from Kaggle (https://www.kaggle.com/datasets/ehababoelnaga/diabetes-dataset/data). The model was trained on standardized features and achieved a best test accuracy of 81%.

## Visualization

The confusion matrix of the best model is visualized using Matplotlib and Seaborn, showing the number of true positives, true negatives, false positives, and false negatives.

![Confusion Matrix](/confusion_matrix.png)

## License

This project is licensed under the MIT License - see the LICENSE file for details.





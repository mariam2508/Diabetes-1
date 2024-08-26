import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Load the trained models (this assumes you have saved these models as .pkl files)
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM (Linear)': LinearSVC(C=1, loss="hinge"),
    'SVM (Non-linear)': SVC(kernel='poly', C=1, degree=6, coef0=0, probability=True),
    'Decision Tree': DecisionTreeClassifier(max_depth=3)
}

# Title and introduction
st.title('Diabetes Prediction App')
st.write('This app predicts the likelihood of diabetes based on user input using different algorithms.')

# Sidebar for model selection and user input
st.sidebar.header('Select Model and Input Features')

model_choice = st.sidebar.selectbox('Choose a model', list(models.keys()))

def user_input_features():
    Pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    Glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    BMI = st.sidebar.slider('BMI', 0.0, 50.0, 25.0)

    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BMI': BMI
    }
    
    # Use pd.DataFrame to create the DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Display input features
input_df = user_input_features()
st.write('Input features')
st.write(input_df)

# Make predictions using the selected model
model = models[model_choice]
prediction = model.predict(input_df)

# Use predict_proba only if the model supports it
if hasattr(model, 'predict_proba'):
    prediction_proba = model.predict_proba(input_df)
    st.write('Prediction Probability')
    st.write(prediction_proba)
else:
    st.write("Model does not support probability prediction.")

# Display results
st.write(f'Prediction using {model_choice} (0: No Diabetes, 1: Diabetes)')
st.write(prediction[0])

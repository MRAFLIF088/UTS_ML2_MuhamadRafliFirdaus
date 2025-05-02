import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('diabetes_model.h5')

scaler = MinMaxScaler()

st.title("Diabetes Prediction App")

# Input fields for user data
pregnancies = st.number_input("Pregnancies", min_value=0, value=2)
glucose = st.number_input("Glucose", min_value=0, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=25)
insulin = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=28.5)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=0, value=40)


# Create a button to trigger prediction
if st.button("Predict"):
    # Create a DataFrame from the user input
    new_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })

    # Assuming X contains the feature names from your original training data
    X_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    new_data = new_data[X_columns]

    # Scale the new data using the loaded scaler
    new_data_scaled = pd.DataFrame(scaler.fit_transform(new_data), columns=new_data.columns)
    new_data_scaled = new_data_scaled.astype(np.float32)

    # Make the prediction
    predicted_outcome = model.predict(new_data_scaled)
    hasil = (predicted_outcome[0][0] > 0.5).astype(int)

    # Display the prediction
    st.write(f"Prediksi Outcome: {'Diabetes' if hasil == 1 else 'Tidak Diabetes'} (score: {predicted_outcome[0][0]:.4f})")

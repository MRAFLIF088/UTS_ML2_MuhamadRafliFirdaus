import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load the trained TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Assuming scaler and label encoder were saved (replace with your actual saving method if different)
# If you saved scaler using joblib:
# scaler = joblib.load('scaler.pkl')
# If you saved label encoder:
# le = joblib.load('label_encoder.pkl')
# For this example, we'll create dummy ones if not saved
# In a real scenario, you would save these during training
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
scaler = MinMaxScaler()
le = LabelEncoder()
# Fit dummy data to the scaler and label encoder to replicate the training setup
dummy_df = pd.DataFrame(np.random.rand(10, 8), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
scaler.fit(dummy_df)
le.fit([0, 1]) # Assuming binary classification (0 and 1)


st.title('Diabetes Prediction App')

st.write("""
This app predicts the likelihood of a person having diabetes based on their medical information.
Please enter the patient's details below.
""")

# Create input fields for each feature
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, value=0)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=99, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=846, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=67.1, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.42, value=0.5)
age = st.number_input('Age', min_value=21, max_value=81, value=30)

# Create a button to trigger prediction
if st.button('Predict'):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Scale the input data using the loaded or recreated scaler
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = input_data_scaled.astype(np.float32) # TFLite models often expect float32

    # Perform inference with TFLite model
    interpreter.set_tensor(input_details[0]['index'], input_data_scaled)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Interpret the output
    # For a binary classification with softmax, the output is probabilities for each class
    # Assuming class 0 is Non-Diabetes and class 1 is Diabetes
    prediction_prob = output_data[0][1] # Probability of class 1 (Diabetes)

    # Get the predicted class label (0 or 1)
    predicted_class_index = np.argmax(output_data)
    predicted_outcome = le.inverse_transform([predicted_class_index])[0]


    st.subheader('Prediction Results')
    if predicted_outcome == 1:
        st.error(f'Based on the input, the prediction is: **Diabetes**')
    else:
        st.success(f'Based on the input, the prediction is: **Non-Diabetes**')

    st.write(f'Probability of Diabetes: {prediction_prob:.2f}')

st.write("""
**Note:** This is a predictive model and not a substitute for professional medical advice.
""")


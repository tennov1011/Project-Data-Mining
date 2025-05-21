import streamlit as st
import pickle
import numpy as np

# Load model dan scaler
with open('mdl.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pickle', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Mental Health Prediction App")

# Input user
name = st.text_input("Name")

gender = st.number_input("Gender (0=Female, 1=Male)", min_value=0, max_value=1, step=1)
age = st.number_input("Age", min_value=18.0, step=0.1)
city = st.number_input("City (Encoded)", min_value=0, max_value=97, step=1)
working_or_student = st.number_input("Working Professional or Student (0=Student, 1=Working)", min_value=0, max_value=1, step=1)
profession = st.number_input("Profession (Encoded)", min_value=0, max_value=63, step=1)
academic_pressure = st.number_input("Academic Pressure", min_value=1.0, max_value=5.0, step=0.1)
work_pressure = st.number_input("Work Pressure", min_value=1.0, max_value=5.0, step=0.1)
cgpa = st.number_input("CGPA", min_value=5.03, max_value=10.0, step=0.01)
study_satisfaction = st.number_input("Study Satisfaction", min_value=1.0, max_value=5.0, step=0.1)
job_satisfaction = st.number_input("Job Satisfaction", min_value=1.0, max_value=5.0, step=0.1)
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=3.0, step=0.1)
dietary_habits = st.number_input("Dietary Habits (0=Unhealthy, 1=Moderate, 2=Healthy)", min_value=0.0, max_value=2.0, step=1.0)
degree = st.number_input("Degree (Encoded)", min_value=0, max_value=114, step=1)
suicidal_thoughts = st.number_input("Have you ever had suicidal thoughts? (0=No, 1=Yes)", min_value=0, max_value=1, step=1)
work_study_hours = st.number_input("Work/Study Hours", min_value=0.0, max_value=12.0, step=0.1)
financial_stress = st.number_input("Financial Stress", min_value=1.0, max_value=5.0, step=0.1)
family_history = st.number_input("Family History of Mental Illness (0=No, 1=Yes)", min_value=0, max_value=1, step=1)
stress_score = st.number_input("Stress Score", min_value=2.0, max_value=10.0, step=0.1)

# Normalisasi fitur tertentu
scaled_values = scaler.transform([[age, work_study_hours, financial_stress, work_pressure]])
normalized_age, normalized_work_study_hours, normalized_financial_stress, normalized_work_pressure = scaled_values[0]

# Final feature list
input_features = np.array([[ 
    gender,
    normalized_age,                 # Gunakan hasil scaling
    city,
    working_or_student,
    profession,
    academic_pressure,
    normalized_work_pressure,       # Gunakan hasil scaling
    cgpa,
    study_satisfaction,
    job_satisfaction,
    sleep_duration,
    dietary_habits,
    degree,
    suicidal_thoughts,
    normalized_work_study_hours,    # Gunakan hasil scaling
    normalized_financial_stress,    # Gunakan hasil scaling
    family_history,
    stress_score,
]])


if st.button("Predict"):
    prediction = model.predict(input_features)[0]
    result = "Depresi" if prediction == 1 else "Tidak Depresi"
    st.success(f"Hasil Prediksi untuk {name}: {result}")

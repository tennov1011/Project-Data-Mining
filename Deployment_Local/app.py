from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model dan scaler
with open('mdl.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pickle', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form
        raw_age = float(request.form['Age'])
        raw_work_study_hours = float(request.form['Work/Study Hours'])
        raw_financial_stress = float(request.form['Financial Stress'])
        raw_work_pressure = float(request.form['Work Pressure'])

        # Bentuk array untuk transform
        scaled_features = scaler.transform([[raw_age, raw_work_study_hours, raw_financial_stress, raw_work_pressure]])
        normalized_age = scaled_features[0][0]
        normalized_work_study_hours = scaled_features[0][1]
        normalized_financial_stress = scaled_features[0][2]
        normalized_work_pressure = scaled_features[0][3]

        # Ambil input lain tanpa normalisasi
        name = request.form['Name']
        input_values = [
            float(request.form['Gender']),
            normalized_age,
            float(request.form['City']),
            float(request.form['Working Professional or Student']),
            float(request.form['Profession']),
            float(request.form['Academic Pressure']),
            normalized_work_pressure,
            float(request.form['CGPA']),
            float(request.form['Study Satisfaction']),
            float(request.form['Job Satisfaction']),
            float(request.form['Sleep Duration']),
            float(request.form['Dietary Habits']),
            float(request.form['Degree']),
            float(request.form['Have you ever had suicidal thoughts ?']),
            normalized_work_study_hours,
            normalized_financial_stress,
            float(request.form['Family History of Mental Illness']),
            float(request.form['isStudents']),
            float(request.form['isWorkingProfessional']),
            float(request.form['Stress_Score']),
        ]

        features = np.array([input_values])
        prediction = model.predict(features)[0]
        result = "Depresi" if prediction == 1 else "Tidak Depresi"

        return render_template('index.html', prediction_text=f"Hasil Prediksi dari {name}: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Terjadi kesalahan: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)


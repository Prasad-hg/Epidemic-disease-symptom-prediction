from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load('models/malaria_dengue_model.pkl')
gender_encoder = joblib.load('models/gender_encoder.pkl')
location_encoder = joblib.load('models/location_encoder.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')


high_risk_cities = ['Bangalore', 'Delhi', 'Hyderabad', 'Kerala', 'Mumbai']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = request.form['age']
    gender = request.form['gender']
    location = request.form['location']
    fever = int(request.form['fever'])
    headache = int(request.form['headache'])
    joint_pain = int(request.form['joint_pain'])
    muscle_pain = int(request.form['muscle_pain'])
    fatigue = int(request.form['fatigue'])
    nausea_vomiting = int(request.form['nausea_vomiting'])
    rash = int(request.form['rash'])
    chills = int(request.form['chills'])
    abdominal_pain = int(request.form['abdominal_pain'])
    bleeding = int(request.form['bleeding'])
    symptom_duration = int(request.form['symptom_duration'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    rainfall = float(request.form['rainfall'])


    gender_encoded = gender_encoder.transform([gender])[0]
    location_encoded = location_encoder.transform([location])[0]

 
    user_input = {
        'Age': age,
        'Gender': gender_encoded,
        'Location': location_encoded,
        'Fever': fever,
        'Headache': headache,
        'Joint Pain': joint_pain,
        'Muscle Pain': muscle_pain,
        'Fatigue': fatigue,
        'Nausea/Vomiting': nausea_vomiting,
        'Rash': rash,
        'Chills': chills,
        'Abdominal Pain': abdominal_pain,
        'Bleeding': bleeding,
        'Symptom Duration (Days)': symptom_duration,
        'Temperature (°C)': temperature,
        'Humidity (%)': humidity,
        'Rainfall (mm)': rainfall
    }

    user_input_df = pd.DataFrame([user_input])

   
    prediction = model.predict(user_input_df)
    probability = model.predict_proba(user_input_df).max(axis=1)[0] * 100

   
    disease = prediction[0]
    city = location  

    severe_symptoms = [
        symptom for symptom in ['Fever', 'Joint Pain', 'Fatigue', 'Chills']
        if user_input[symptom] == 3
    ]

    if 25 <= temperature <= 35:
        temperature_risk = "High temperature, conducive for mosquito activity."
    elif 20 <= temperature < 25:
        temperature_risk = "Moderate temperature, somewhat conducive for mosquito activity."
    else:
        temperature_risk = "Low temperature, less favorable for mosquito activity."

    humidity_risk = (
        "High humidity, favorable for mosquito breeding." if humidity > 80 else
        "Moderate humidity, somewhat conducive for mosquito breeding." if humidity > 60 else
        "Low humidity, less favorable for mosquito breeding."
    )

    rainfall_risk = (
        "High rainfall, indicating possible mosquito breeding." if rainfall > 50 else
        "Moderate rainfall, with potential for mosquito breeding." if rainfall > 20 else
        "Low rainfall, unlikely to support mosquito breeding."
    )

    high_risk_city = city in high_risk_cities
    
    return render_template(
        'result.html',
        disease=disease,
        probability=f"{probability:.2f}",
        city=city,
        high_risk_city=high_risk_city,
        severe_symptoms=', '.join(severe_symptoms) if severe_symptoms else 'None',
        temperature_risk=temperature_risk,
        humidity_risk=humidity_risk,
        rainfall_risk=rainfall_risk
    )

if __name__ == '__main__':
    app.run(debug=True)
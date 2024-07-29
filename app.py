from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('models/rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def get_recommendation(risk_level, probability):
    if risk_level == 'High':
        return f"Based on the prediction, there is a high risk of diabetes (probability: {probability:.2%}). It is important to consult with a healthcare provider for further evaluation and management. Adopting a healthy lifestyle, such as a balanced diet and regular exercise, can help mitigate the risk."
    else:
        return f"Based on the prediction, there is a low risk of diabetes (probability: {probability:.2%}). Continue to maintain a healthy lifestyle to keep the risk low."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        gender = float(request.form['gender'])
        age = float(request.form['age'])
        hypertension = float(request.form['hypertension'])
        heart_disease = float(request.form['heart_disease'])
        smoking_history = float(request.form['smoking_history'])
        bmi = float(request.form['bmi'])
        hba1c_level = float(request.form['hba1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])

        
        # Prepare features for prediction
        features = np.array([gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level])
        
        # Scale the input features
        input_features = scaler.transform([features])
        
        # Make prediction
        probability = model.predict_proba(input_features)[0][1]
        risk_level = 'High' if probability > 0.5 else 'Low'
        
        # Get recommendation based on prediction
        recommendation = get_recommendation(risk_level, probability)
        
        return render_template('index.html', result=risk_level, probability=f"{probability:.2%}", recommendation=recommendation)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


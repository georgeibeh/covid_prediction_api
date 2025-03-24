from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
model_filename = 'covid_model.pkl'
with open(model_filename, 'rb') as model_file:
    model = joblib.load('covid_model.pkl')

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    df = pd.read_csv(filepath)
    
    # Ensure columns match expected input format
    expected_columns = [
        "Sex", "Chest pain", "Chills or sweats", "Confused or disoriented", "Cough", "Diarrhea", 
        "Difficulty breathing or Dyspnea", "Pain behind eyes or Sensitivity to light", 
        "Fatigue or general weakness", "Fever", "Fluid in the lung cavity", 
        "Fluid in lung cavity in auscultation", "Fluid in cavity through X-Ray", "Bleeding of the gums", 
        "Headache", "Joint pain or arthritis", "Thorax (sore throat)", "Malaise", "Muscle pain", 
        "Nausea", "Other clinical symptoms", "Pharyngeal exudate", "Rapid breathing", "Runny nose", 
        "Maculopapular rash", "Sore throat or pharyngitis", "Bleeding or bruising", "Vomiting", 
        "Abnormal lung X-Ray findings", "Conjunctivitis", "Acute respiratory distress syndrome", 
        "Pneumonia (clinical or radiologic)", "Loss of Taste", "Loss of Smell", "Cough with sputum", 
        "Cough with heamoptysis", "Enlarged lymph nodes", "Wheezing", "Skin ulcers", "Inability to walk", 
        "Indrawing of chest wall", "Other complications", "Age"
    ]
    
    if not all(col in df.columns for col in expected_columns):
        return "Uploaded CSV does not match expected format", 400
    
    X = df[expected_columns]  # Select the relevant features
    predictions = model.predict(X)
    df['COVID_Prediction'] = predictions
    
    output_filepath = os.path.join(OUTPUT_FOLDER, 'predictions.csv')
    df.to_csv(output_filepath, index=False)
    
    return send_file(output_filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

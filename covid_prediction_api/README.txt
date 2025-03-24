COVID-19 Prediction API

This Flask-based API takes a CSV file containing patient symptoms and predicts the presence of COVID-19 using a pre-trained machine learning model.

Features

    Upload CSV files via a simple web interface.

    Model processes input and appends predictions.

    Download the updated CSV with predictions.

Installation & Setup

Clone the repository:

     git clone https://github.com/georgeibeh/covid_prediction_api.git
     cd covid_prediction_api

Install dependencies:

    pip install -r requirements.txt

Run the Flask app:

    python app.py

Open in your browser:

    http://127.0.0.1:5000/

Expected CSV Format

  Ensure your CSV has the following columns:

    Sex, Chest pain, Chills or sweats, Confused or disoriented, Cough, ..., Age

Missing or incorrect columns may result in errors.

Deployment

    Deploy with Gunicorn (Production)

    pip install gunicorn
    gunicorn -w 4 -b 0.0.0.0:5000 app:app

Deploy on Heroku

 Install Heroku CLI and login:

    heroku login

Deploy:

   git init
   heroku create
   git add .
   git commit -m "Initial commit"
   git push heroku master


# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)

# Load your pre-trained AI model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    # input_data = [, request.form['feature2']]  # Adjust based on your features
    data = {
    'Description_Length': request.form['feature1'],
    'Priority': request.form['feature2'],
    'Technical_Complexity': request.form['feature3']
    }
    custom_data = pd.DataFrame(data)
    print("custom data",custom_data)

    # Preprocess input data if necessary

    # Make predictions using your AI model
    prediction = model.predict(custom_data)
    print("predictions",prediction)

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

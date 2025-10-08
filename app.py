
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))  # Fixed: load scaler once

@app.route('/')
def home():
    return render_template('home.html')  # Ensure you have a home.html in /templates

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        # Get JSON data
        data = request.json['data']  # e.g., {"data": {"feature1": value1, "feature2": value2, ...}}
        print("Raw input:", data)

        # Convert to numpy array and reshape
        input_array = np.array(list(data.values())).reshape(1, -1)
        print("Input array:", input_array)

        # Scale the data
        scaled_input = scaler.transform(input_array)

        # Predict using the model
        prediction = model.predict(scaled_input)
        print("Prediction:", prediction[0])

        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)

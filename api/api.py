from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model 
model = joblib.load("models/random_forest_03-03-2025-23-25-43.pkl")  

@app.route('/predict', methods=['POST'])
def predict():
    # Expecting JSON input
    data = request.get_json(force=True)
    # Convert JSON to DataFrame – assume the JSON is a dict of feature values or list of dicts
    df_input = pd.DataFrame(data if isinstance(data, list) else [data])
    # Predict using the pipeline
    predictions = model.predict(df_input)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

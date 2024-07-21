"""
Asthma Prediction Flask App
"""

import pickle
from io import StringIO
import pandas as pd
from flask import Flask, request, jsonify

app = Flask('asthma-prediction')

def preprocessing(df):
    """
    Preprocess the input DataFrame.

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Processed data.
    """
    columns = ['DustExposure', 'GastroesophagealReflux', 'LungFunctionFEV1',
                'LungFunctionFVC', 'Wheezing', 'ChestTightness', 'Coughing',
                'NighttimeSymptoms', 'ExerciseInduced']
    
    return df[columns].fillna(value=0)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict asthma probability based on input data.

    Returns:
        flask.Response: JSON response containing predicted probabilities.
    """
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    json_data = request.get_json()
    df = pd.read_json(StringIO(json_data), orient="index")
    df = df.T
    X_test = preprocessing(df)
    result = jsonify(model.predict(X_test).tolist())
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

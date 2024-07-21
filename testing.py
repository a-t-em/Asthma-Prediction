import json
import requests

def test_prediction():
    """
    Test the prediction endpoint of the Flask app.

    Modify the test_case dictionary to experiment with different input data.
    """
    # The URL of the Flask app (replace with your actual URL)
    url = 'http://127.0.0.1:5000/predict'

    # Example data (modify to experiment)
    test_case = {
        'DustExposure': 5,
        'GastroesophagealReflux': 0.1,
        'LungFunctionFEV1': 2,
        'LungFunctionFVC': 4,
        'Wheezing': 0.6,
        'ChestTightness': 0.5,
        'Coughing': 0.5,
        'NighttimeSymptoms': 0.6,
        'ExerciseInduced': 0.6
    }

    json_data = json.dumps(test_case)

    # Send the POST request
    response = requests.post(url, json=json_data, timeout=5)  # Add a timeout (e.g., 5 seconds)

    assert 0 <= response.json()[0] <= 1

if __name__ == '__main__':
    test_prediction()

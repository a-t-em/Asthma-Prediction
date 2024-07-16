import pytest
import json
import scoring_script
from scoring_script import predict

test_case = {'DustExposure': 5, 'GastroesophagealReflux': 0.1, 'LungFunctionFEV1': 2,
            'LungFunctionFVC': 4, 'Wheezing': 0.6, 'ChestTightness': 0.5, 'Coughing': 0.5,
            'NighttimeSymptoms': 0.6, 'ExerciseInduced': 0.6}
json_string = json.dumps(test_case)
result = predict(json_string)

assert 0 <= result <= 1
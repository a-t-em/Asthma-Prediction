import pickle
import pandas as pd
import numpy as np
from io import StringIO

def preprocessing(df):
    columns = ['DustExposure', 'GastroesophagealReflux', 'LungFunctionFEV1',
                'LungFunctionFVC', 'Wheezing', 'ChestTightness', 'Coughing',
                'NighttimeSymptoms', 'ExerciseInduced']
    
    return df[columns].fillna(value=0)

def predict(json_object):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_json(StringIO(json_object), orient="index")
    df = df.T
    X_test = preprocessing(df)
    return model.predict(X_test)


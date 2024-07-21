"""
Asthma Prediction Model

This script preprocesses data, creates a random forest model for asthma prediction,
and performs hyperparameter optimization using Hyperopt and MLflow.

Usage:
    - Ensure the input data file 'asthma_disease_data.csv' is available in the './data' directory.
    - To utilize MLflow, launch MLflow with 'mlflow ui' command in the terminal.
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
import mlflow 
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope

def get_data(df):
    """
    Preprocesses the data by dropping irrelevant columns and handling missing values.

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        tuple: Processed features and target.
    """
    features = df.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis=1)
    target = df['Diagnosis']
    
    correlation_matrix = features.corrwith(target)

    threshold = 0.02
    columns_to_drop = correlation_matrix[abs(correlation_matrix) < threshold].index
    features = features.drop(columns_to_drop, axis=1)
    features = features.fillna(value=0)
    
    return features, target
    
def create_model(data_path="./data/asthma_disease_data.csv"):
    """
    Creates a random forest model for asthma prediction using hyperparameter optimization.

    Args:
        data_path (str, optional): Path to the input data CSV file. Defaults to "./data/asthma_disease_data.csv".
    """
    df = pd.read_csv(data_path)
    features, target = get_data(df)

    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("asthma_prediction")

    smote = SMOTETomek(random_state=42)
    X, y = smote.fit_resample(features, target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "rf")
            mlflow.log_params(params)

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)

            mlflow.sklearn.log_model(model, "random_forest_model")
            mlflow.log_metric("validation_score -- accuracy", accuracy)

            return {'loss': accuracy, 'status': STATUS_OK, 'Trained_Model': model}
        
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # Set seed value for reproducibility
    trials = Trials()

    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials,
        rstate=rstate
    )

    best_trial_index = np.argmin([r['loss'] for r in trials.results])
    best_model = trials.results[best_trial_index]['Trained_Model']

    with open("model.pkl", "wb") as file:
        pickle.dump(best_model, file) 
    
    X_test['prediction'] = best_model.predict(X_test)
    X_train['prediction'] = best_model.predict(X_train)
    
    outdir = './data'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    pred_file = os.path.join(outdir, 'test_data')    
    X_test.to_csv(pred_file, index=False)  

    train_file = os.path.join(outdir, 'training_data') 
    X_train.to_csv(train_file, index=False)

if __name__ == '__main__':
    create_model()
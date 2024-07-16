#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.kaggle.com/code/yeemeitsang/asthma-prediction?scriptVersionId=187483634" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a>

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from imblearn.combine import SMOTETomek
import mlflow 
import pickle

def get_data(data_path):
    df = pd.read_csv(data_path)

    features = df.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis=1)
    target = df['Diagnosis']

    correlation_matrix = features.corrwith(target)

    threshold = 0.02
    columns_to_drop = correlation_matrix[abs(correlation_matrix) < threshold].index
    features = features.drop(columns_to_drop, axis=1)

    return features, target

def create_model(data_path="asthma_disease_data.csv"):
    features, target = get_data(data_path)

    # To utilize mlflow, first launch mflow with mlflow ui command in the terminal
    # Then create an experiment in the UI and plug in experiment_id as a parameter
    mlflow.set_experiment(experiment_id="836971815908963715")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    with mlflow.start_run():
        mlflow.autolog()

        smote = SMOTETomek(random_state=42)
        X, y = smote.fit_resample(features, target)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.log_metric("validation_score -- accuracy", accuracy)

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)   

if __name__ == '__main__':
    create_model()
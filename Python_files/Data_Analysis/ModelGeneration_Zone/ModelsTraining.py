import duckdb
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from Python_files.Data_Analysis.ModelGeneration_Zone.NumericalFeatureNeuralNetwork import NumericalFeatureNeuralNetwork
import pandas as pd
import numpy as np
import joblib
import torch
import os

def create_model_folder():
    if not os.path.exists('./Data Analysis/ModelGeneration Zone/Models'):
        try:
            os.makedirs('./Data Analysis/ModelGeneration Zone/Models')
        except Exception as e:
            print('Error: Could not create folder for models.')
            return False
    return True

def create_model(X_train, y_train, model_name):
    try:
        if model_name == 'SGDLinearRegressor':
            modelRegressor = SGDRegressor(random_state=42)
        elif model_name == 'RandomForestRegressor':
            modelRegressor = RandomForestRegressor(random_state=42)
        elif model_name == 'MLPRegressor':
            modelRegressor = MLPRegressor(random_state=42)
        else:
            modelRegressor = NumericalFeatureNeuralNetwork(random_state=42, hidden_layers=(100,))

        model = modelRegressor.fit(X_train, y_train)

        y_pred = modelRegressor.predict(X_train)
        r2 = r2_score(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)

        return (True, r2, mse, modelRegressor)
    
    except Exception as e:
        print(f'Error: Could not create model {model_name}.')
        return (False,)

def save_model(model, model_name):
    try:
        if model_name == 'nfnnRegressorModel':
            filename = f'./Data Analysis/ModelGeneration Zone/Models/{model_name}.pth'
            torch.save(model, filename)
        else:
            filename = f'./Data Analysis/ModelGeneration Zone/Models/{model_name}.pkl'
            joblib.dump(model, filename)
            
        print(f'Model saved as {filename}')
    except Exception as e:
        print(f'Error: Could not save model {filename}.')
        return False
    return True


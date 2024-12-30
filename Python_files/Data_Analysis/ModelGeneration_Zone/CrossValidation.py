import duckdb
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from Python_files.Data_Analysis.ModelGeneration_Zone.NumericalFeatureNeuralNetwork import NumericalFeatureNeuralNetwork
from sklearn.model_selection import cross_validate, ShuffleSplit
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import os


def create_cross_validation_folder():
    try:
        if not os.path.exists('Data Analysis/ModelGeneration Zone/Models/CrossValidation'):  
            os.makedirs('Data Analysis/ModelGeneration Zone/Models/CrossValidation/', exist_ok=True)
            os.makedirs('Data Analysis/ModelGeneration Zone/Models/CrossValidation/SGDLinearRegressorModel/', exist_ok=True)
            os.makedirs('Data Analysis/ModelGeneration Zone/Models/CrossValidation/RandomForestRegressorModel/', exist_ok=True)
            os.makedirs('Data Analysis/ModelGeneration Zone/Models/CrossValidation/MLPRegressorModel/', exist_ok=True)
            os.makedirs('Data Analysis/ModelGeneration Zone/Models/CrossValidation/NFNNRegressorModel/', exist_ok=True)
        return True
    except Exception as e:
        print('Error: Could not create folder for cross-validation.')
        return False
    
def cross_validation(X, y):
    models = {
        'sgdLinearRegressorModel': SGDRegressor(random_state=42),
        'randomForestRegressorModel': RandomForestRegressor(random_state=42),
        'mlpRegressorModel': MLPRegressor(random_state=42),
        'nfnnRegressorModel': NumericalFeatureNeuralNetwork(random_state=42, hidden_layers=(100,))
    }

    scoring = {
        'r2': 'r2',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'explained_variance' : 'explained_variance'
    }

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    model_cv_results = {}
    try:
        for model_name, model in models.items():
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False, return_estimator=True, return_indices=True)
            model_cv_results[model_name] = cv_results
    except Exception as e:
        print(f'Error: Could not performance Cross Validation')
        return (False,)
    return (True, model_cv_results)

def save_split_sets(cv_results, X, y, con):
    try:
        train_indices = cv_results['indices']['train']
        test_indices = cv_results['indices']['test']

        for i in range(len(train_indices)):
            X_train_indices = X.iloc[train_indices[i]]
            y_train_indices = y.iloc[train_indices[i]]

            X_test_indices = X.iloc[test_indices[i]]
            y_test_indices = y.iloc[test_indices[i]]

            train_set = pd.concat([X_train_indices, y_train_indices], axis=1)
            test_set = pd.concat([X_test_indices, y_test_indices], axis=1)

            con.execute(f"DROP TABLE IF EXISTS cross_validation_train_set_{i}")
            con.execute(f"DROP TABLE IF EXISTS cross_validation_test_set_{i}")
            con.execute(f'CREATE TABLE IF NOT EXISTS cross_validation_train_set_{i} AS SELECT * FROM train_set')
            con.execute(f'CREATE TABLE IF NOT EXISTS cross_validation_test_set_{i} AS SELECT * FROM test_set')
            print(f'Fold {i+1} training and test sets stored')

        return True
    except Exception as e:
        print(f'Error: Could not save split sets into database')
        return False
            
        

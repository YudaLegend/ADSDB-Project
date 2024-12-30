from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV, ShuffleSplit
from Python_files.Data_Analysis.ModelGeneration_Zone.NumericalFeatureNeuralNetwork import NumericalFeatureNeuralNetwork
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def hyperparameter_tuning(X, y):
    nfnnRegressorModel = NumericalFeatureNeuralNetwork(random_state=42)

    param_grid = {
        'hidden_layers': [(100,), (64, 32, 16), (128, 64, 32, 16)],
        'dropout_rate': [0.1, 0.2, 0.3],
        'lr': [0.001, 0.003, 0.005],
        'wd': [0.0001, 0.0003, 0.0005],
        'batch_size': [32, 64, 128],
    }

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    halvingGridSearch = HalvingGridSearchCV(estimator=nfnnRegressorModel, param_grid=param_grid, scoring='r2', 
                        n_jobs=1, cv=cv, return_train_score=True, factor=3, random_state=42)
    try:
        halving_grid_search = halvingGridSearch.fit(X, y)
        print(f'Best hyperparameters: {halving_grid_search.best_params_}')
        print(f'Best R2 score: {halving_grid_search.best_score_}')

        return (True, halving_grid_search)
    except Exception as e:
        print(f'Error: Could not perform hyperparameter tuning')
        return (False,)

    
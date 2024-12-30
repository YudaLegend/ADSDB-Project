from Python_files.Data_Analysis.ModelGeneration_Zone.NumericalFeatureNeuralNetwork import NumericalFeatureNeuralNetwork
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
import numpy as np
import pandas as pd
import torch
import os

def metric_results(y_test, y_pred):
    return r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred), explained_variance_score(y_test, y_pred)

def predictor_target_split(data, type):
    if type == 1:
        X = data[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
        'recommendation_ratio', 'average_playtime', 'median_playtime', 'price_discount']]
        
    else:
        X = data[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
        'recommendation_ratio']]
    y = data['game_satisfaction']
    return X, y

def create_cv_feature_selection_folder():
    try:
        if not os.path.exists('Data Analysis/ModelGeneration Zone/Models/CrossValidation/NFNNRegressorModel_FeatureSelection'):  
            os.makedirs('Data Analysis/ModelGeneration Zone/Models/CrossValidation/NFNNRegressorModel_FeatureSelection/', exist_ok=True)
        return True
    except Exception as e:
        print(f'Error: Could not create folder for cross-validation feature selection NFNNRegressor model. {e}')
        return False
    
def nfnnRegressor_feature_predict(nfnnRegressor, X_test, y_test):
    try:
        y_pred = nfnnRegressor.predict(X_test)
        y_pred = y_pred.ravel().tolist()
        r2, mse, explained_variance = metric_results(y_test, y_pred)
        return (True, r2, mse, explained_variance, y_pred)
    
    except Exception as e:
        print(f"Error in nfnnRegressor_feature_predict: {e}")
        return (False,)
    
def split_datasets(con):
    try:
        split_datasets = []
        for i in range(5):
            test = con.execute(f'SELECT * FROM cross_validation_test_set_{i}').df()
            train = con.execute(f'SELECT * FROM cross_validation_train_set_{i}').df()
            split_datasets.append((train, test))
        return split_datasets
    except Exception as e:
        print(f"Error in split_datasets: {e}")
        return []

def split_datasets_nfnn_results(split_datasets):
    r2_score_nfnnRegressor = {}
    mse_score_nfnnRegressor = {}
    explained_variance_score_nfnnRegressor = {}
    try:
        for i, (train, test) in enumerate(split_datasets):
            nfnnRegressor_estimator = torch.load(f"Data Analysis/ModelGeneration Zone/Models/CrossValidation/NFNNRegressorModel/nfnnRegressorModel_cross_validation_estimator_{i}.pth")
            
            X_test, y_test = predictor_target_split(test, 1)

            y_pred = nfnnRegressor_estimator.predict(X_test)
            r2, mse, explained_variance = metric_results(y_test, y_pred)

            r2_score_nfnnRegressor[f'estimator_{i}'] = r2
            mse_score_nfnnRegressor[f'estimator_{i}'] = mse
            explained_variance_score_nfnnRegressor[f'estimator_{i}'] = explained_variance
        
        return (True, r2_score_nfnnRegressor, mse_score_nfnnRegressor, explained_variance_score_nfnnRegressor)
    
    except Exception as e:
        print(f"Error in cross_validation_nfnn_: {e}")
        return (False, )
    
def cross_validation_nfnn_feature(split_datasets):
    r2_score_nfnnRegressor_feature_selection = {}
    mse_score_nfnnRegressor_feature_selection = {}
    explained_variance_score_nfnnRegressor_feature_selection = {}

    try:
        for i, (train, test) in enumerate(split_datasets):
            nfnnRegressor_feature_selection_estimator = NumericalFeatureNeuralNetwork(random_state=42, hidden_layers=(100,)) 

            X_train, y_train = predictor_target_split(train, -1)
            X_test, y_test = predictor_target_split(test, -1)

            nfnnRegressor_feature_selection_estimator.fit(X_train, y_train)
            
            y_pred = nfnnRegressor_feature_selection_estimator.predict(X_test)
            r2, mse, explained_variance = metric_results(y_test, y_pred)

            r2_score_nfnnRegressor_feature_selection[f'estimator_{i}'] = r2
            mse_score_nfnnRegressor_feature_selection[f'estimator_{i}'] = mse
            explained_variance_score_nfnnRegressor_feature_selection[f'estimator_{i}'] = explained_variance

            torch.save(nfnnRegressor_feature_selection_estimator, f"Data Analysis/ModelGeneration Zone/Models/CrossValidation/nfnnRegressorModel_FeatureSelection/nfnnRegressorModel_FeatureSelection_cross_validation_estimator_{i}.pth")
            print(f'Mode saved as Data Analysis/ModelGeneration Zone/Models/CrossValidation/nfnnRegressorModel_FeatureSelection/nfnnRegressorModel_FeatureSelection_cross_validation_estimator_{i}.pth')
        return (True, r2_score_nfnnRegressor_feature_selection, mse_score_nfnnRegressor_feature_selection, explained_variance_score_nfnnRegressor_feature_selection)
    
    except Exception as e:
        print(f"Error in cross_validation_nfnn_feature: {e}")
        return (False, )
        
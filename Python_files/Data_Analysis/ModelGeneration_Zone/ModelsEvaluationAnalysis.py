import joblib
from sklearn.metrics import r2_score, root_mean_squared_error
from matplotlib import pyplot as plt
import torch

def load_model(model_path):
    return joblib.load(model_path)

def metric_results(y_test, y_pred):
    return r2_score(y_test, y_pred), root_mean_squared_error(y_test, y_pred)


def model_predict(model, model_name, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        if model_name == 'nfnnRegressorModel':
            y_pred = y_pred.ravel().tolist()
        r2_score, RMSE = metric_results(y_test, y_pred)
        return (True, r2_score, RMSE, y_pred)
    
    except Exception as e:
        print(f"Error in {model_name} predict: {e}")
        return (False,)

def frequency_distribution(y_test, y_preds):

    sgdLinearRegressor_y_pred = y_preds[0]
    randomForestRegressor_y_pred = y_preds[1]
    mlpRegressor_y_pred = y_preds[2]
    nfnnRegressor_y_pred = y_preds[3]

    sgdLinearRegressor_count = 0
    randomForestRegressor_count = 0
    mlpRegressor_count = 0
    nfnnRegressor_count = 0
    for i in range(len(y_test)):
        sgdLinearRegressor_pred = abs(sgdLinearRegressor_y_pred[i] - y_test[i])
        randomForestRegressor_pred = abs(randomForestRegressor_y_pred[i] - y_test[i])
        mlpRegressor_pred = abs(mlpRegressor_y_pred[i] - y_test[i])
        nfnnRegressor_pred = abs(nfnnRegressor_y_pred[i] - y_test[i])

        if sgdLinearRegressor_pred < randomForestRegressor_pred and sgdLinearRegressor_pred < mlpRegressor_pred and sgdLinearRegressor_pred < nfnnRegressor_pred:
            sgdLinearRegressor_count += 1
        elif randomForestRegressor_pred < sgdLinearRegressor_pred and randomForestRegressor_pred < mlpRegressor_pred and randomForestRegressor_pred < nfnnRegressor_pred:
            randomForestRegressor_count += 1
        elif mlpRegressor_pred < sgdLinearRegressor_pred and mlpRegressor_pred < randomForestRegressor_pred and mlpRegressor_pred < nfnnRegressor_pred:
            mlpRegressor_count += 1
        else:
            nfnnRegressor_count += 1

    frequency_results = {}
    frequency_results['sgdLinearRegressorModel'] = sgdLinearRegressor_count
    frequency_results['randomForestRegressorModel'] = randomForestRegressor_count
    frequency_results['mlpRegressorModel'] = mlpRegressor_count
    frequency_results['nfnnRegressorModel'] = nfnnRegressor_count

    return frequency_results


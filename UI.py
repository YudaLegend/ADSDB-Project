import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as messagebox
import duckdb
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os

# Importing the functions from the Fist Part of the project
from Python_files.Data_Management.Landing_Zone import TemporalToPersistent
from Python_files.Data_Management.Formatted_Zone import PersistentToFormatted
from Python_files.Data_Management.Trusted_Zone import FormattedToTrusted, TrustedQualityDataCleaning, TrustedQualityExtractionTransformation, TrustedQualityCurrencyConversion, TrustedQualityTranslation
from Python_files.Data_Management.Exploitation_Zone import TrustedToExploitation, ExploitationQualityKPI, ExploitationTables

#Importing the functions from the Second Part of the project
from Python_files.Data_Analysis.AnalyticalSandbox_Zone import ExploitationToAnalyticalSandbox
from Python_files.Data_Analysis.FeatureEngineering_Zone.Feature_Generation import FeatureGenerationConvertToNumerical, FeatureGenerationConvertToCategorical,FeatureGenerationCreateFeature

from Python_files.Data_Analysis.FeatureEngineering_Zone.Data_Preparation import DataPreparationOutlierMissingData,DataPreparationEncoding

from Python_files.Data_Analysis.FeatureEngineering_Zone import GenerationTrainTest 

from Python_files.Data_Analysis.ModelGeneration_Zone import ModelsTraining, ModelsEvaluationAnalysis, CrossValidation, FeatureSelection, FeatureSelectionAnalysis, HyperparameterTuning


#Importing others functions
from Python_files.Monitoring.monitor_script import start_monitoring_thread, stop_monitoring_thread
from Python_files.Check_Code import CheckCodeQuality



def connectToDatabase(db_path):
    try:
        con = duckdb.connect(database=db_path, read_only=False)
    except Exception as e:
        write_to_terminal('Error connecting to the database with path'+ db_path +'and error message:'+ str(e)+'\n')
        return None
    return con

def write_to_terminal(message):
    terminal_text.config(state=tk.NORMAL)  # Enable text widget to insert new text
    terminal_text.insert(tk.END, message + "\n")
    terminal_text.see(tk.END)  # Automatically scroll to the bottom
    terminal_text.config(state=tk.DISABLED)  # Disable editing again
    root.update()

def write_to_terminal2(message):
    terminal_text2.config(state=tk.NORMAL)
    terminal_text2.insert(tk.END, message + "\n")
    terminal_text2.see(tk.END)
    terminal_text2.config(state=tk.DISABLED)
    root.update()

def plot_bar_results(results, x_label, y_label, title):
    plt.figure(figsize=(10, 6))
    color = ['red', 'blue', 'green', 'purple']
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.bar(list(results.keys()), list(results.values()), color=color)
    plt.show()


def plot_results(results, x_label, y_label, title):
    plt.figure(figsize=(10, 6))
    for k, v in results.items():
        plt.plot(list(v.keys()), list(v.values()), label=k, marker='o')
    plt.legend()
    plt.xticks(rotation=45, ha="right", fontsize=8)  # Rotate and align labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

######################### Data Management Functions #########################
def temporalToPersistent():
    root.after(500)
    TemporalToPersistent.create_folders()
    write_to_terminal("All the zone folders created !!!\n")

    root.after(500)
    succefull = TemporalToPersistent.move_files()
    if succefull:
        write_to_terminal("Completed the file Management !!!\n")
        return True
    else:
        write_to_terminal("Error in file management !!!\n")
        return False

def persistentToFormatted():
    root.after(500)
    con = connectToDatabase('./Data Management/Formatted Zone/formatted_zone.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the database !!!\n")
        return False
    write_to_terminal("Connected to the database !!!\n")

    root.after(500)
    datasetsFormatted = PersistentToFormatted.formattedDatasets(con)
    if datasetsFormatted:
        write_to_terminal("Completed the formatted datasets management !!!\n")
    else:
        write_to_terminal("Error in formatted datasets management !!!\n")
        
    con.close()
    write_to_terminal("Disconnected from the database to save memory !!!\n")

    if datasetsFormatted:
        return True
    else:
        return False

def formattedToTrusted():
    formatted_con = connectToDatabase('./Data Management/Formatted Zone/formatted_zone.duckdb')
    if formatted_con is None:
        write_to_terminal("Error connecting to the formatted database !!!\n")
        return False
    write_to_terminal("Connected to the formatted database !!!\n")
    trusted_conn = connectToDatabase('./Data Management/Trusted Zone/trusted_zone.duckdb')
    if trusted_conn is None:
        write_to_terminal("Error connecting to the trusted database !!!\n")
        return False
    write_to_terminal("Connected to the trusted database !!!\n")


    write_to_terminal("Starting the cleaning process ...\n")

    root.after(500)
    epic_tables_to_combine = ['epic_games_v1', 'epic_games_v2']
    epicUnifyTable = FormattedToTrusted.UnifyTables(epic_tables_to_combine,"epic_games", formatted_con, trusted_conn)
    if epicUnifyTable:
        write_to_terminal("Completed the Epic Games dataset unification !!!\n")
    else:
        write_to_terminal("Error in Epic Games dataset unification !!!\n")

    root.after(500)
    steam_spy_tables_to_combine = ['steam_spy_v1','steam_spy_v2']
    steamSpyUnifyTable = FormattedToTrusted.UnifyTables(steam_spy_tables_to_combine,"steam_spy", formatted_con, trusted_conn)
    if steamSpyUnifyTable:
        write_to_terminal("Completed the SteamSpy dataset unification !!!\n")
    else:
        write_to_terminal("Error in SteamSpy dataset unification !!!\n")

    root.after(500)
    steam_players_to_combine = ['steam_current_players_v1','steam_current_players_v2']
    steamPlayersUnifyTable = FormattedToTrusted.UnifyTables(steam_players_to_combine,"steam_current_players", formatted_con, trusted_conn)
    if steamPlayersUnifyTable:
        write_to_terminal("Completed the Steam current players dataset unification !!!\n")
    else:
        write_to_terminal("Error in Steam current players dataset unification !!!\n")

    root.after(500)
    steam_game_info_to_combine = ['steam_app_details_v1','steam_app_details_v2']    
    steamGameUnifyTable = FormattedToTrusted.UnifyTables(steam_game_info_to_combine,"steam_app_details", formatted_con, trusted_conn)
    if steamGameUnifyTable:
        write_to_terminal("Completed the Steam App Details dataset unification !!!\n")
    else:
        write_to_terminal("Error in Steam App Details dataset unification !!!\n")

    trusted_conn.execute("DROP VIEW combined_data_df;")
    formatted_con.close()
    trusted_conn.close()
    write_to_terminal('Disconnected from the trusted database and formatted database to save memory !!!\n')

    if epicUnifyTable and steamSpyUnifyTable and steamPlayersUnifyTable and steamGameUnifyTable:
        return True
    else:
        return False

def trustedQuality():
    con = connectToDatabase('./Data Management/Trusted Zone/trusted_zone.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the trusted database !!!\n")
        return False
    write_to_terminal('Connected to the trusted database for quality assessment!!!\n')

    root.after(500)
    data_cleaning = TrustedQualityDataCleaning.data_cleaning(con)
    if data_cleaning:
        write_to_terminal("Completed the data cleaning process !!!\n")
    else:
        write_to_terminal("Error in data cleaning process !!!\n")

    
    root.after(500)
    data_extraction_transformation = TrustedQualityExtractionTransformation.data_extraction_transformation(con)
    if data_extraction_transformation:
        write_to_terminal("Completed the data extraction and transformation process !!!\n")
    else:
        write_to_terminal("Error in data extraction and transformation process !!!\n")

    
    root.after(500)
    data_currency_conversion = TrustedQualityCurrencyConversion.data_currency_conversion(con)
    if data_currency_conversion:
        write_to_terminal("Completed the currency conversion process !!!\n")
    else:
        write_to_terminal("Error in currency conversion process !!!\n")


    root.after(500)
    data_translation = TrustedQualityTranslation.data_translation(con)
    if data_translation:
        write_to_terminal("Completed the data translation process !!!\n")
    else:
        write_to_terminal("Error in data translation process !!!\n")
    
    con.close()
    write_to_terminal('Disconnected from the trusted database to save memory !!!\n')
    if data_cleaning and data_extraction_transformation and data_currency_conversion and data_translation:
        return True
    return False

def trustedToExploitation():
    root.after(500)
    trusted_con = connectToDatabase('./Data Management/Trusted Zone/trusted_zone.duckdb')
    if trusted_con is None:
        write_to_terminal("Error connecting to the trusted database !!!\n")
        return False
    write_to_terminal('Connected to the trusted database !!!\n')

    exploitation_conn = connectToDatabase('./Data Management/Exploitation Zone/exploitation_zone.duckdb')
    if exploitation_conn is None:
        write_to_terminal("Error connecting to the exploitation database !!!\n")
        return False
    write_to_terminal('Connected to the exploitation database !!!\n')
    

    root.after(500)
    tablesMerged = TrustedToExploitation.mergeTables(trusted_con, exploitation_conn)
    if tablesMerged:
        write_to_terminal("Completed the merging of the datasets !!!\n")
    else: 
        write_to_terminal('Error in merging the datasets !!!\n')

    exploitation_conn.close()
    trusted_con.close()
    write_to_terminal('Disconnected from the trusted database and exploitation database to save memory !!!\n')
    return tablesMerged

def exploitationQuality():
    con = connectToDatabase('./Data Management/Exploitation Zone/exploitation_zone.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the exploitation database !!!\n")
        return False
    write_to_terminal('Connected to the exploitation database for quality assessment !!!\n')

    root.after(500)
    mergeTableQuality = ExploitationQualityKPI.mergeTableQuality(con)
    if mergeTableQuality:
        write_to_terminal("Completed the dataset quality assessment !!!\n")
    else:
        write_to_terminal("Error in dataset quality assessment !!!\n")

    con.close()
    write_to_terminal('Disconnected from the exploitation database to save memory!!!\n')
    return mergeTableQuality


def exploitationTables():
    con = connectToDatabase('./Data Management/Exploitation Zone/exploitation_zone.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the exploitation database !!!\n")
        return False
    write_to_terminal('Connected to the exploitation database for table creation !!!\n')

    root.after(500)
    createdTables = ExploitationTables.createTable(con, "videogame_companies", ["developers"])
    if createdTables:
        write_to_terminal("Completed the table creation !!!\n")
    else:
        write_to_terminal("Error in table creation !!!\n")

    con.close()
    write_to_terminal('Disconnected from the exploitation database to save memory !!!\n')

    return createdTables

def exploitationKPI():
    con = connectToDatabase('./Data Management/Exploitation Zone/exploitation_zone.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the exploitation database !!!\n")
        return False
    write_to_terminal('Connected to the exploitation database for KPI creation !!!\n')

    root.after(500)
    createdKPIsTable = ExploitationQualityKPI.createKPIsTable(con)
    if createdKPIsTable:
        write_to_terminal("Completed the KPI creation table !!!\n")
    else:
        write_to_terminal("Error in KPI creation table !!!\n")

    con.close()
    write_to_terminal('Disconnected from the exploitation database to save memory !!!\n')
    return createdKPIsTable
######################### Data Management Functions #########################


######################### Data Analysis Functions #########################

def analyticalSandbox():
    root.after(500)
    exploitation_con = connectToDatabase('./Data Management/Exploitation Zone/exploitation_zone.duckdb')
    if exploitation_con is None:
        write_to_terminal("Error connecting to the exploitation database !!!\n")
        return False
    write_to_terminal("Connected to the exploitation database !!!\n")
    sandbox_con = connectToDatabase('./Data Analysis/AnalyticalSandbox Zone/analytical_sandbox.duckdb')
    if sandbox_con is None:
        write_to_terminal("Error connecting to the analytical sandbox database !!!\n")
        return False
    write_to_terminal("Connected to the analytical sandbox database !!!\n")

    df = exploitation_con.execute('SELECT * FROM steam_games_kpi').df()
    df.drop(columns=['release_date', 'developers', 'publishers'], inplace=True)

    sandbox_con = connectToDatabase('./Data Analysis/AnalyticalSandbox Zone/analytical_sandbox.duckdb')    
    sandbox_con.execute(f"CREATE TABLE IF NOT EXISTS sandbox_steam_games_kpi AS SELECT * FROM df")

    exploitation_con.close()
    sandbox_con.close()
    write_to_terminal("Disconnected from the exploitation database to save memory !!!\n")
    write_to_terminal("Disconnected from the analytical_sandbox database to save memory !!!\n")
    return sandbox_con is not None and exploitation_con is not None

def featureGenerationConvertToNumerical():
    root.after(500)
    # Connect to the analytical sandbox database
    sandbox_con = connectToDatabase('./Data Analysis/AnalyticalSandbox Zone/analytical_sandbox.duckdb')
    if sandbox_con is None:
        write_to_terminal("Error connecting to the analytical sandbox database !!!\n")
        return False
    write_to_terminal("Connected to the analytical sandbox database !!!\n")

    # Connect to the feature engineering database
    feature_con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if feature_con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    root.after(500)
    featureGenerationCN = FeatureGenerationConvertToNumerical.featureGenerationConvertToNumerical(sandbox_con, feature_con)
    if featureGenerationCN:
        write_to_terminal("Completed the feature generation and convert to numerical process !!!\n")
    else:
        write_to_terminal("Error in feature generation and convert to numerical process !!!\n")

    sandbox_con.close()
    feature_con.close()

    write_to_terminal("Disconnected from the exploitation database to save memory !!!\n")
    write_to_terminal("Disconnected from the analytical_sandbox database to save memory !!!\n")
    return featureGenerationCN

def featureGenerationConvertToCategorical():
    root.after(500)
    # Connect to the feature engineering database
    feature_con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if feature_con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    root.after(500)
    featureGenerationCC = FeatureGenerationConvertToCategorical.featureGenerationConvertToCategorical(feature_con)
    if featureGenerationCC:
        write_to_terminal("Completed the feature generation and convert to categorical process !!!\n")
    else:
        write_to_terminal("Error in feature generation and convert to categorical process !!!\n")

    feature_con.close()
    write_to_terminal("Disconnected from the feature database to save memory !!!\n")
    return featureGenerationCC

def featureGenerationCreateFeature():
    root.after(500)
    # Connect to the feature engineering database
    feature_con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if feature_con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    root.after(500)
    featureGenerationCF = FeatureGenerationCreateFeature.featureGenerationCreateFeature(feature_con)
    if featureGenerationCF:
        write_to_terminal("Completed the feature generation create feature process !!!\n")
    else:
        write_to_terminal("Error in feature generation create feature process !!!\n")

    feature_con.close()
    write_to_terminal("Disconnected from the feature database to save memory !!!\n")
    return featureGenerationCF

def dataPreparationOutlierMissingData():
    root.after(500)
    # Connect to the feature engineering database
    feature_con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if feature_con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    root.after(500)
    dataPreparationOMD = DataPreparationOutlierMissingData.dataPreparationOutlierMissingData(feature_con)

    if dataPreparationOMD:
        write_to_terminal("Completed the data preparation outlier missing data process !!!\n")
    else:
        write_to_terminal("Error in data preparation outlier missing data process !!!\n")

    feature_con.close()
    write_to_terminal("Disconnected from the feature database to save memory !!!\n")
    return dataPreparationOMD

def dataPreparationEncoding():
    root.after(500)
    # Connect to the feature engineering database
    feature_con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if feature_con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    root.after(500)
    dataPreparationE = DataPreparationEncoding.dataPreparationEncoding(feature_con)

    if dataPreparationE:
        write_to_terminal("Completed the data preparation encoding process !!!\n")
    else:
        write_to_terminal("Error in data preparation encoding process !!!\n")

    feature_con.close()
    write_to_terminal("Disconnected from the feature database to save memory !!!\n")
    return dataPreparationE

def generationTrainTest():
    root.after(500)
    # Connect to the feature engineering database
    feature_con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if feature_con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    root.after(500)
    generationTT = GenerationTrainTest.generationTrainTest(feature_con)

    if generationTT:
        write_to_terminal("Completed the generation train test process !!!\n")
    else:
        write_to_terminal("Error in generation train test process !!!\n")

    feature_con.close()
    write_to_terminal("Disconnected from the feature database to save memory !!!\n")
    return generationTT


def trainModelResults(model, model_name, estimator_name):
    if model[0]:
        if estimator_name == 'nfnnRegressorModel' or estimator_name == 'nfnnRegressorModel_feature_selection':
            torch.save(model[3], f'./Data Analysis/ModelGeneration Zone/Models/{estimator_name}.pth')
        else:
            ModelsTraining.save_model(model[3], estimator_name)
    else:
        write_to_terminal(f"Error in {model_name} model training process !!!\n")
        return False
    write_to_terminal(f'{model_name} model Train scores -> r2 score: {model[1]}, MSE: {model[2]}')
    write_to_terminal(f"Completed the {model_name} model training process !!!\n")

def modelsTraining():
    root.after(500)
    # Connect to the feature engineering database
    feature_con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if feature_con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    root.after(500)

    isCreate = ModelsTraining.create_model_folder()
    if isCreate:
        write_to_terminal("Completed the model folder creation process !!!\n")
    else:
        write_to_terminal("Error in model folder creation process !!!\n")
        return False

    df = feature_con.execute('SELECT * FROM train_dataset').df()
    X_train = df[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
        'recommendation_ratio', 'average_playtime', 'median_playtime', 'price_discount']]
    y_train = df['game_satisfaction'].tolist()

    root.after(500)
    sgdRegressorResults = ModelsTraining.create_model(X_train, y_train, 'SGDLinearRegressor')
    trainModelResults(sgdRegressorResults, 'SGD Linear Regressor', 'sgdLinearRegressorModel')

    root.after(500)
    randomForestRegressorResults = ModelsTraining.create_model(X_train, y_train, 'RandomForestRegressor')
    trainModelResults(randomForestRegressorResults, 'Random Forest Regressor', 'randomForestRegressorModel')

    root.after(500)
    mlpRegressorResults = ModelsTraining.create_model(X_train, y_train, 'MLPRegressor')
    trainModelResults(mlpRegressorResults, 'MLP Regressor', 'mlpRegressorModel')

    root.after(500)
    nfnnRegressorResults = ModelsTraining.create_model(X_train, y_train, 'NFNNRegressor')
    trainModelResults(nfnnRegressorResults, 'Numerical Feature Neural Network Regressor', 'nfnnRegressorModel')

    feature_con.close()
    write_to_terminal("Disconnected from the feature database to save memory !!!\n")

    return sgdRegressorResults[0] and randomForestRegressorResults[0] and mlpRegressorResults[0] and nfnnRegressorResults[0]


def testModelResults(modelResults, model_name, estimator_name, r2_results, rmse_results, y_preds):
    if modelResults[0]:
        r2_results[estimator_name] = modelResults[1]
        rmse_results[estimator_name] = modelResults[2]
        y_preds.append(modelResults[3])
        write_to_terminal(f'{model_name} model Test scores -> r2 score: {modelResults[1]}, RMSE: {modelResults[2]}')
        write_to_terminal(f"Completed the {model_name} model evaluation process !!!\n")
    else: 
        write_to_terminal(f"Error in {model_name} model evaluation process !!!\n")

def max_model_result(results, metric):
    max_key = max(results, key=results.get)
    max_value = results[max_key]
    print(f"Best model: {max_key} with {metric} score: {max_value}")
    write_to_terminal(f"Best model: {max_key} with {metric} score: {max_value}")

def modelsEvaluationAnalysis():
    root.after(500)
    # Connect to the feature engineering database
    feature_con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if feature_con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    root.after(500)
    df = feature_con.execute('SELECT * FROM test_dataset').df()
    X_test = df[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
        'recommendation_ratio', 'average_playtime', 'median_playtime', 'price_discount']]
    y_test = df['game_satisfaction'].tolist()
    
    r2_results = {}
    rmse_results = {}
    y_preds = []

    root.after(500)
    sgdRegressorModel = ModelsEvaluationAnalysis.load_model('./Data Analysis/ModelGeneration Zone/Models/sgdLinearRegressorModel.pkl')
    sgdRegressorResults = ModelsEvaluationAnalysis.model_predict(sgdRegressorModel, 'sgdRegressorModel', X_test, y_test)
    testModelResults(sgdRegressorResults, 'SGD Linear Regressor', 'sgdLinearRegressorModel', r2_results, rmse_results, y_preds)

    root.after(500)
    randomForestRegressorModel = ModelsEvaluationAnalysis.load_model('./Data Analysis/ModelGeneration Zone/Models/randomForestRegressorModel.pkl')
    randomForestRegressorResults = ModelsEvaluationAnalysis.model_predict(randomForestRegressorModel, 'randomForestRegressorModel', X_test, y_test)
    testModelResults(randomForestRegressorResults, 'Random Forest Regressor', 'randomForestRegressorModel', r2_results, rmse_results, y_preds)

    root.after(500)
    mlpRegressorModel = ModelsEvaluationAnalysis.load_model('./Data Analysis/ModelGeneration Zone/Models/mlpRegressorModel.pkl')
    mlpRegressorResults = ModelsEvaluationAnalysis.model_predict(mlpRegressorModel, 'mlpRegressorModel', X_test, y_test)
    testModelResults(mlpRegressorResults, 'MLP Regressor', 'mlpRegressorModel', r2_results, rmse_results, y_preds)

    root.after(500)
    nfnnRegressorModel = torch.load('./Data Analysis/ModelGeneration Zone/Models/nfnnRegressorModel.pth')
    nfnnRegressorResults = ModelsEvaluationAnalysis.model_predict(nfnnRegressorModel, 'nfnnRegressorModel', X_test, y_test)
    testModelResults(nfnnRegressorResults, 'Numerical Feature Neural Network Regressor', 'nfnnRegressorModel', r2_results, rmse_results, y_preds)

    root.after(500)
    frequency_results = ModelsEvaluationAnalysis.frequency_distribution(y_test, y_preds)
    plot_bar_results(frequency_results, 'Model Name', 'Frequency (Number of observations)', 'Model Frequency Counter')
    max_model_result(frequency_results, 'Counter Metric')

    plot_bar_results(r2_results, 'Model Name', 'R2 Score', 'R2 Score Results')
    max_model_result(r2_results, 'R2 Metric')

    plot_bar_results(rmse_results, 'Model Name', 'RMSE Score', 'RMSE Score Results')
    neg_rmse_results = {key: -value for key, value in rmse_results.items()}
    max_model_result(neg_rmse_results, 'Negative RMSE Metric')

    total = sum(frequency_results.values())
    frequency_results = {key: value / total for key, value in frequency_results.items()}
    plot_results({'R2 Metric': r2_results, "RMSE Metric": rmse_results, "Normalized Counter Metric": frequency_results}, 'Model Name', 'Metric Scores', 'Model Evaluation Metrics')

    feature_con.close()
    write_to_terminal("Disconnected from the feature database to save memory !!!\n")

    return sgdRegressorResults[0] and randomForestRegressorResults[0] and mlpRegressorResults[0] and nfnnRegressorResults[0]

def crossValidationResults(cross_validation_results, r2_results, neg_mse_results, explained_variance_results, fit_time_results):
    for model_name, cv_results in cross_validation_results[1].items():
        mean_r2 = np.mean(cv_results['test_r2'])
        mean_neg_mse = np.mean(cv_results['test_neg_mean_squared_error'])
        explained_variance = np.mean(cv_results['test_explained_variance'])
        fit_time = np.mean(cv_results['fit_time'])
        write_to_terminal(f"    {model_name}: R^2 = {mean_r2:.4f}, Neg MSE = {mean_neg_mse:.4f}, Explained Variance = {explained_variance:.4f}, Fit Time = {fit_time:.4f}\n")
        r2_results[model_name] = mean_r2
        neg_mse_results[model_name] = mean_neg_mse
        explained_variance_results[model_name] = explained_variance
        fit_time_results[model_name] = fit_time

        for estimator in list(enumerate(cv_results['estimator'])):
            if model_name == 'nfnnRegressorModel':
                torch.save(estimator[1], f"Data Analysis/ModelGeneration Zone/Models/CrossValidation/{model_name}/{model_name}_cross_validation_estimator_{estimator[0]}.pth")
                print(f'Model saved as Data Analysis/ModelGeneration Zone/Models/CrossValidation/{model_name}/{model_name}_cross_validation_estimator_{estimator[0]}.pth')
            else:
                joblib.dump(estimator[1], f"Data Analysis/ModelGeneration Zone/Models/CrossValidation/{model_name}/{model_name}_cross_validation_estimator_{estimator[0]}.pkl")
                print(f'Model saved as Data Analysis/ModelGeneration Zone/Models/CrossValidation/{model_name}/{model_name}_cross_validation_estimator_{estimator[0]}.pkl')

def cross_validation():
    root.after(500)
    # Connect to the feature engineering database
    con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    root.after(500)
    df = con.execute("SELECT * FROM feature_steam_games").df()
    X = df[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
        'recommendation_ratio', 'average_playtime', 'median_playtime', 'price_discount']]
    y = df['game_satisfaction']

    isCreated = CrossValidation.create_cross_validation_folder()
    if not isCreated:
        write_to_terminal("Error creating the cross validation folder !!!\n")
        return False
    write_to_terminal("Cross Validation Folders Created !!!\n")

    root.after(500)
    cross_validation_results = CrossValidation.cross_validation(X, y)
    if not cross_validation_results[0]:
        write_to_terminal("Error performing cross validation !!!\n")
        return False
    write_to_terminal("Cross Validation Process Completed Successfully !!!\n")

    root.after(500)
    write_to_terminal(f"Cross Validation Results:")

    neg_mse_results, r2_results, explained_variance_results, fit_time_results = {}, {}, {}, {}
    crossValidationResults(cross_validation_results, r2_results, neg_mse_results, explained_variance_results, fit_time_results)

    root.after(500)
    split_sets_saved = CrossValidation.save_split_sets(cross_validation_results[1]['nfnnRegressorModel'], X, y, con)
    if not split_sets_saved:
        write_to_terminal("Error saving split sets !!!\n")
        return False
    write_to_terminal("Split Sets Saved Successfully !!!\n")

    root.after(500)
    mse_results = {key: abs(value) for key, value in neg_mse_results.items()}
    plot_bar_results(r2_results, 'Model Name', 'R2 Score', 'Average R2 Score of Cross-Validation Comparison')
    max_model_result(r2_results, 'R2')

    plot_bar_results(mse_results, 'Model Name', 'MSE Score', 'Average MSE of Cross-Validation Comparison')
    max_model_result(neg_mse_results, 'Negative MSE')
    
    plot_bar_results(explained_variance_results, 'Model Name', 'Explained Variance Score', 'Average Explained Variance Score of Cross-Validation Comparison')
    max_model_result(explained_variance_results, 'Explained Variance')

    plot_results({'Fit Time': fit_time_results}, 'Model Name', 'Time (seconds)', 'Average Fit Time of Cross-Validation Comparison')
    plot_results({'MSE Metric': mse_results, 'R2 Score Metric': r2_results, 'Explained Variance Metric': explained_variance_results}, 'Model Name', 'Metric Values', 'Cross Validation Results')
    
    write_to_terminal('Best Model is Numerical Features Neural Network !!!\n')

    con.close()
    write_to_terminal("Disconnected from the feature database to save memory !!!\n")
    return True

def feature_selection():
    root.after(500)
    # Connect to the feature engineering database
    con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    root.after(500)
    df = con.execute("SELECT * FROM feature_steam_games").df()
    X = df[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
       'recommendation_ratio', 'average_playtime','median_playtime', 'price_discount']]
    y = df['game_satisfaction']

    varianceThreshold_results = FeatureSelection.varianceThreshold(X)
    if varianceThreshold_results[0]:
        write_to_terminal(f'Variance Threshold Variablles: {varianceThreshold_results[1]}\n')

    rRegression_results = FeatureSelection.rRegression(X, y)
    if rRegression_results[0]:
        write_to_terminal(f"R Regression Significant features: {rRegression_results[1]}\n")

    fRegression_results = FeatureSelection.fRegression(X, y)
    if fRegression_results[0]:
        write_to_terminal(f"F Regression Significant features: {fRegression_results[1]}\n")

    con.close()
    write_to_terminal("Disconnected from the feature database !!!\n")
    return varianceThreshold_results[0] and rRegression_results[0] and fRegression_results[0]

def feature_selection_nfnn():
    root.after(500)
    # Connect to the feature engineering database
    con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    df = con.execute("SELECT * FROM train_dataset").df()
    X_train = df[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
        'recommendation_ratio']]
    y_train = df['game_satisfaction'].tolist()

    root.after(500)
    nfnnRegressorResults = ModelsTraining.create_model(X_train, y_train, 'nfnnRegressor')
    trainModelResults(nfnnRegressorResults, 'Numerical Feature Neural Network Regressor With Feature Selection', 'nfnnRegressorModel_feature_selection')

    con.close()
    write_to_terminal("Disconnected from the feature database !!!\n")

    return nfnnRegressorResults[0]

def featureSelectionModelResults(nfnnRegressorResults, model_name, estimator_name, r2_results, mse_results, explained_variance_results):
    if nfnnRegressorResults[0]:
        r2_results[estimator_name] = nfnnRegressorResults[1]
        mse_results[estimator_name] = nfnnRegressorResults[2]
        explained_variance_results[estimator_name] = nfnnRegressorResults[3]
        write_to_terminal(f'{model_name} model Test scores -> r2 score: {nfnnRegressorResults[1]}, MSE: {nfnnRegressorResults[2]}, Explained Variance: {nfnnRegressorResults[3]}')
        write_to_terminal(f"Completed the {model_name} model evaluation process !!!\n")
    else: 
        write_to_terminal(f"Error in {model_name} model evaluation process !!!\n")

def feature_selection_nfnn_analysis():
    root.after(500)
    # Connect to the feature engineering database
    con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    isCreated = FeatureSelectionAnalysis.create_cv_feature_selection_folder()
    if not isCreated:
        write_to_terminal("Error creating the CV feature selection folder !!!\n")
        return False
    write_to_terminal("CV feature selection folders created successfully !!!\n")

    root.after(500)
    test = con.execute("SELECT * FROM test_dataset").df()
    data = con.execute("SELECT * FROM feature_steam_games").df()

    X_test = test[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
        'recommendation_ratio', 'average_playtime', 'median_playtime', 'price_discount']]
    X_test_feature_selection = test[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
            'recommendation_ratio']]
    y_test = test['game_satisfaction'].tolist()

    nfnnRegressor = torch.load("Data Analysis/ModelGeneration Zone/Models/nfnnRegressorModel.pth")
    nfnnRegressor_feature_selection = torch.load("Data Analysis/ModelGeneration Zone/Models/nfnnRegressorModel_feature_selection.pth")

    r2_score_results, mse_score_results, explained_variance_score_results = {}, {}, {}
    root.after(500)
    nfnnRegressorModel = FeatureSelectionAnalysis.nfnnRegressor_feature_predict(nfnnRegressor, X_test, y_test)
    featureSelectionModelResults(nfnnRegressorModel, 'NFNN Regressor', 'nfnnRegressorModel', r2_score_results, mse_score_results, explained_variance_score_results)
    
    root.after(500)
    nfnnRegressorFeatureSelectionModel = FeatureSelectionAnalysis.nfnnRegressor_feature_predict(nfnnRegressor_feature_selection, X_test_feature_selection, y_test)
    featureSelectionModelResults(nfnnRegressorFeatureSelectionModel, 'NFNN Regressor With Feature Selection', 'nfnnRegressorModel_feature_selection', r2_score_results, mse_score_results, explained_variance_score_results)

    plot_results({'R2 Score': r2_score_results, 'Explained Variance Score': explained_variance_score_results}, 'NFNN Model Type', 'Metric Scores', 'Model Metric Scores Analysis')
    plot_results({'MSE Score': mse_score_results}, 'NFNN Model Type', 'MSE Score', 'Model MSE Score Analysis')

    split_datasets = FeatureSelectionAnalysis.split_datasets(con)
    if len(split_datasets) == 0:
        write_to_terminal("Error in getting split datasets for cross-validation. Please check the terminal output for more details.\n")
        return False
    write_to_terminal("Get split datasets successfully for cross-validation !!!\n")

    root.after(500)
    nfnnRegressorCVResults = FeatureSelectionAnalysis.split_datasets_nfnn_results(split_datasets)
    nfnnRegressorFeatureSelectionCVResults = FeatureSelectionAnalysis.cross_validation_nfnn_feature(split_datasets)

    if nfnnRegressorCVResults[0] and nfnnRegressorFeatureSelectionCVResults[0]:
        write_to_terminal("Completed the NFNN Regressor Model with Feature Selection Cross-Validation...\n")
        write_to_terminal("Completed the NFNN Regressor Model Cross-Validation...\n")

        plot_results({'without feature selection': nfnnRegressorCVResults[1], 'with feature selection': nfnnRegressorFeatureSelectionCVResults[1]}, 'Estimator of NFNN Regressor Model', 'R2 Score', 'R2 Score Analysis of NFNN Regressor Models')
        plot_results({'without feature selection': nfnnRegressorCVResults[2], 'with feature selection': nfnnRegressorFeatureSelectionCVResults[2]}, 'Estimator of NFNN Regressor Model', 'MSE Score', 'MSE Score Analysis of NFNN Regressor Models')
        plot_results({'without feature selection': nfnnRegressorCVResults[3], 'with feature selection': nfnnRegressorFeatureSelectionCVResults[3]}, 'Estimator of NFNN Regressor Model', 'Explained Variance Score', 'Explained Variance Score Analysis of NFNN Regressor Models')

        r2_score_mean = np.mean(list(nfnnRegressorCVResults[1].values()))
        r2_score_feature_selection_mean = np.mean(list(nfnnRegressorFeatureSelectionCVResults[1].values()))

        mse_score_mean = np.mean(list(nfnnRegressorCVResults[2].values()))
        mse_score_feature_selection_mean = np.mean(list(nfnnRegressorFeatureSelectionCVResults[2].values()))

        explained_variance_score_mean = np.mean(list(nfnnRegressorCVResults[3].values()))
        explained_variance_score_feature_selection_mean = np.mean(list(nfnnRegressorFeatureSelectionCVResults[3].values()))

        plot_results({'R2 Score': {'without feature selection': r2_score_mean, 'with feature selection' : r2_score_feature_selection_mean}, 'Explained Variance Score': {'without feature selection' : explained_variance_score_mean, 'with feature selection' : explained_variance_score_feature_selection_mean}}, 'NFNN Regressor Model', 'Metric Scores', 'Average Metric Analysis for NFNN Regressor Model')
        plot_results({'MSE Score': {'without feature selection' : mse_score_mean, 'with feature selection' : mse_score_feature_selection_mean}}, 'NFNN Regressor Model', 'MSE Score', 'Average MSE Analysis for NFNN Regressor Model')

    con.close()
    write_to_terminal("Database connection closed successfully...\n")

    return nfnnRegressorModel[0] and nfnnRegressorFeatureSelectionModel[0] and nfnnRegressorCVResults[0] and nfnnRegressorFeatureSelectionCVResults[0]


def getSearchResults(halving_grid_search):
    search_results = pd.DataFrame(halving_grid_search.cv_results_)
    search_results['iter'] = search_results['iter'].astype(str)
    search_results['n_resources'] = search_results['n_resources'].astype(str)
    search_results['param_batch_size'] = search_results['param_batch_size'].astype(str)
    search_results['param_dropout_rate'] = search_results['param_dropout_rate'].astype(str)
    search_results['param_lr'] = search_results['param_lr'].astype(str)
    search_results['param_wd'] = search_results['param_wd'].astype(str)
    search_results['param_hidden_layers'] = search_results['param_hidden_layers'].astype(str)
    return search_results

def finalIterationsTop10Plots(search_results):
    last_3_iterations = sorted(search_results['iter'].unique())[-3:]
    filtered_results = search_results[search_results['iter'].isin(last_3_iterations)]
    final_iterations_top10 = filtered_results.sort_values(by='mean_test_score', ascending=False).head(10)
    final_iterations_top10 = final_iterations_top10.iloc[::-1]
    mean_fit_time_results = final_iterations_top10['mean_fit_time']
    mean_test_score_results = final_iterations_top10['mean_test_score']
    mean_train_score_results = final_iterations_top10['mean_train_score']
    index = 'iter='+final_iterations_top10['iter']+'\n'+'n_resources='+final_iterations_top10['n_resources']+'\n'+'batch_size:'+final_iterations_top10['param_batch_size']+'\n'+'dropout_rate:'+final_iterations_top10['param_dropout_rate']+'\n'+'hidden_layers:'+final_iterations_top10['param_hidden_layers']+'\n'+'learning_rate:'+final_iterations_top10['param_lr']+'\n'+'weight_decay:'+final_iterations_top10['param_wd']

    mean_fit_time_results = dict(zip(index, mean_fit_time_results))
    mean_test_score_results = dict(zip(index, mean_test_score_results))
    mean_train_score_results = dict(zip(index, mean_train_score_results))

    plot_results({'Mean Test Score': mean_test_score_results, 'Mean Train Score': mean_train_score_results}, 'Hyperparameter Combinations', 'R2 Score', "Mean Train and Test Scores for Top 10 Hyperparameters Combinations (Last iterations)")
    plot_results({'Mean Fit Time': mean_fit_time_results}, 'Hyperparameter Combinations', 'Mean Fit Time', "Mean Fit Time for Top 10 Hyperparameters Combinations (Last iterations)")


def bestParamsPlots(search_results, best_params):
    best_params_iteration = search_results[search_results['params'] == best_params]
    mean_fit_time_results = best_params_iteration['mean_fit_time']
    mean_test_score_results = best_params_iteration['mean_test_score']
    mean_train_score_results = best_params_iteration['mean_train_score']
    index = 'iter='+best_params_iteration['iter']+'\n'+'n_resources='+best_params_iteration['n_resources']

    mean_fit_time_results = dict(zip(index, mean_fit_time_results))
    mean_test_score_results = dict(zip(index, mean_test_score_results))
    mean_train_score_results = dict(zip(index, mean_train_score_results))

    plot_results({'Mean Test Score': mean_test_score_results, 'Mean Train Score': mean_train_score_results}, 'Iteration + Resource', 'R2 Score', "Mean Train and Test Scores for Best Hyperparameter Combination")
    plot_results({'Mean Fit Time': mean_fit_time_results}, 'Iteration + Resource', 'Mean Fit Time', "Mean Fit Time for the Best Hyperparameters Combination")

def hyperparameter_tuning():
    root.after(500)
    # Connect to the feature engineering database
    con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    df = con.execute("SELECT * FROM feature_steam_games").df()
    X = df[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
        'recommendation_ratio']]
    y = df['game_satisfaction'].tolist()

    write_to_terminal('Start Hyperparameter Tuning Process...\n')
    hyperparameter_tuning_results = HyperparameterTuning.hyperparameter_tuning(X, y)
    if not hyperparameter_tuning_results[0]:
        write_to_terminal("Failed to perform Hyperparameter Tuning. Please check the terminal output for more details.\n")
        return False
    write_to_terminal("Hyperparameter Tuning completed successfully\n")

    write_to_terminal(f"Best Hyperparameters: {hyperparameter_tuning_results[1].best_params_}\n")
    write_to_terminal(f"Best Score: {hyperparameter_tuning_results[1].best_score_}\n")

    search_results = getSearchResults(hyperparameter_tuning_results[1])

    finalIterationsTop10Plots(search_results)
    bestParamsPlots(search_results, hyperparameter_tuning_results[1].best_params_)

    torch.save(hyperparameter_tuning_results[1].best_estimator_, f'./Data Analysis/ModelGeneration Zone/Models/nfnnRegressorModel_best_estimator_with_feature_selection.pth')
    print(f'Model saved as ./Data Analysis/ModelGeneration Zone/Models/nfnnRegressorModel_best_estimator_with_feature_selection.pth')
    con.close()
    write_to_terminal("Database connection closed successfully...\n")

    return hyperparameter_tuning_results[0]

def button3_action():
    button3.config(state=tk.DISABLED)
    write_to_terminal("Starting the Data Analysis...\n")
    write_to_terminal("Loading data...\n")
    
    write_to_terminal("Create Folders for the Data Analysis...\n")
    successful = ExploitationToAnalyticalSandbox.create_folders()

    if not successful:
        write_to_terminal("Failed to create the folders for the Data Analysis. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Folders Created for the Data Analysis...\n")

    successful = analyticalSandbox()
    
    if not successful:
        write_to_terminal("Failed to create Analytical Sandbox. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Sandbox_analytical process completed successfully !!!\n")

    successful = featureGenerationConvertToNumerical()

    if not successful:
        write_to_terminal("Failed to complete Feature Generation Convert To Numerical. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Feature Generation Convert To Numerical process completed successfully !!!\n")

    successful = featureGenerationConvertToCategorical()

    if not successful:
        write_to_terminal("Failed to complete Feature Generation Convert To Categorical. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Feature Generation Convert To Categorical process completed successfully !!!\n")

    successful = featureGenerationCreateFeature()

    if not successful:
        write_to_terminal("Failed to complete Feature Generation Create Feature. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Feature Generation Create Feature process completed successfully !!!\n")

    successful = dataPreparationOutlierMissingData()
    if not successful:
        write_to_terminal("Failde to complete Data Preparation Outlier Missing Data. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Data Preparation Outlier Missing Data process completed successfully !!!\n")

    successful = dataPreparationEncoding()
    if not successful:
        write_to_terminal("Failed to complete Data Preparation Encoding. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Data Preparation Encoding process completed successfully !!!\n")

    successful = generationTrainTest()
    if not successful:
        write_to_terminal("Failed to generate train/test sets. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Generation Train Test process completed successfully !!!\n")

    successful = modelsTraining()
    if not successful:
        write_to_terminal("Failed to train the models. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Models Training process completed successfully !!!\n")

    successful = modelsEvaluationAnalysis()
    if not successful:
        write_to_terminal("Failed to evaluate and analyze the models. Please check the terminal output for more details.\n")
        return

    successful = cross_validation()
    if not successful:
        write_to_terminal("Failed to perform cross validation. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Cross Validation process completed successfully !!!\n")

    successful = feature_selection()
    if not successful:
        write_to_terminal("Failed to perform feature selection. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Feature Selection process completed successfully !!!\n")

    successful = feature_selection_nfnn()
    if not successful:
        write_to_terminal("Failed to perform feature selection for the Numerical Feature Neural Network Regressor. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Feature Selection for the Numerical Feature Neural Network Regressor process completed successfully !!!\n")
    
    successful = feature_selection_nfnn_analysis()
    if not successful:
        write_to_terminal("Failed to analyze feature selection for the Numerical Feature Neural Network Regressor. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Feature Selection Analysis for the Numerical Feature Neural Network Regressor process completed successfully !!!\n")
   
    successful = hyperparameter_tuning()
    if not successful:
        write_to_terminal("Failed to perform hyperparameter tuning. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Analysis: Hyperparameter Tuning process completed successfully !!!\n")

    write_to_terminal("All processes completed successfully !!!\n")
    write_to_terminal("Data Analysis process completed successfully !!!\n")
    messagebox.showinfo('Data Analysis', 'Data Analysis completed !!!')
    button4.config(state=tk.NORMAL)
    write_to_terminal("Click the 'Start Prediction' button to start the prediction process with the best model :)\n")
    

def button1_action():

    button1.config(state=tk.DISABLED)
    write_to_terminal("Starting the Data Management...\n")
    write_to_terminal("Loading data...\n")

    global monitor_active, monitor_thread
    if not monitor_active:
        write_to_terminal2("Monitoring process started...\n")
        monitor_thread = start_monitoring_thread(write_to_terminal2)  
        monitor_active = True
    else:
        messagebox.showinfo("Execution in progress", "The monitor is already in progress.")

    sucessful = temporalToPersistent()
    if not sucessful: 
        write_to_terminal("Failed to organize the data. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Management: Temporal to Persistent process completed successfully !!!\n")

    sucessful = persistentToFormatted()
    if not sucessful:
        write_to_terminal("Failed to format the data. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Management: Persistent to Formatted process completed successfully !!!\n")
    
    sucessful = formattedToTrusted()
    if not sucessful:
        write_to_terminal("Failed to transform the data. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Management: Formatted to Trusted process completed successfully !!!\n")

    sucessful = trustedQuality()
    if not sucessful:
        write_to_terminal("Failed to assess the quality of the data. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Management: Trusted Quality process completed successfully !!!\n")

    sucessful = trustedToExploitation()
    if not sucessful:
        write_to_terminal("Failed to merge the data. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Management: Trusted to Exploitation process completed successfully !!!\n")

    sucessful = exploitationQuality()
    if not sucessful:
        write_to_terminal("Failed to assess the quality of the merged data. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Management: Exploitation Quality process completed successfully !!!\n")

    sucessful = exploitationKPI()
    if not sucessful:
        write_to_terminal("Failed to create the KPIs. Please check the terminal output for more details.\n")
        return
    write_to_terminal("Data Management: KPIs created successfully !!!\n")

    sucessful = exploitationTables()
    if not sucessful:
        write_to_terminal("Failed to create the tables. Please check the terminal output for more details.\n")
        return
    
    write_to_terminal("All processes completed successfully !!!\n")
    root.after(500)
    write_to_terminal("Data Management process completed successfully !!!\n")
    button3.config(state=tk.NORMAL)
    messagebox.showinfo('Data Management', 'Data Management completed !!!')
    write_to_terminal("Click the 'Start Data Analysis' button to start the data analysis process :)\n")


def convert_test_set_to_csv():
    root.after(500)
    # Connect to the feature engineering database
    con = connectToDatabase('./Data Analysis/FeatureEngineering Zone/feature_engineering.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the feature database !!!\n")
        return False
    write_to_terminal("Connected to the feature database !!!\n")

    root.after(500)
    write_to_terminal("Converting test set to CSV...\n")
    for i in range(5):
        test_set = con.execute(f'SELECT * FROM cross_validation_test_set_{i}').df()
        test_set.to_csv(f'./Data Analysis/Test Set/cross_validation_test_set_{i}.csv', index=False)
    
    test_set = con.execute('SELECT * FROM test_dataset').df()
    test_set.to_csv(f'./Data Analysis/Test Set/test_set.csv', index=False)
    write_to_terminal("Converted test sets to CSV successfully !!!\n")
    con.close()
    write_to_terminal("Database connection closed successfully !!!\n")
    global test_set_created 
    test_set_created = True

test_set_created = False

def button4_action():

    root.after(500)
    if not test_set_created:
        if not os.path.exists('./Data Analysis/Test Set/'):
            os.makedirs('./Data Analysis/Test Set/')
        if not os.path.exists('./Data Analysis/Best Model Predict Set/'):
            os.makedirs('./Data Analysis/Best Model Predict Set/')
        convert_test_set_to_csv()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = filedialog.askopenfilename(
        initialdir=current_directory, 
        title="Select a File",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    print(file_path)
    last_slash_index = file_path.rfind('/')
    if last_slash_index != -1:
        file_name_with_extension = file_path[last_slash_index+1:]
        file_name, _ = os.path.splitext(file_name_with_extension)  # Split off the extension
        print(file_name)

    root.after(500)
    best_model = torch.load('./Data Analysis/ModelGeneration Zone/Models/nfnnRegressorModel_best_estimator_with_feature_selection.pth')
    
    test = pd.read_csv(file_path)
    X_test = test[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
        'recommendation_ratio']]
    y_test = test['game_satisfaction'].tolist()

    pred_results = FeatureSelectionAnalysis.nfnnRegressor_feature_predict(best_model, X_test, y_test)
    write_to_terminal(f'Prediction for the test set: {file_name}')
    write_to_terminal(f'    Predict Score -> R2: {pred_results[1]}, MSE: {pred_results[2]}, Explained Variance: {pred_results[3]}')
    write_to_terminal('Prediction completed successfully !!! \n')
    
        
    pred_results_df = pd.DataFrame(pred_results[4], columns=['predicted_score'])
    pred_results_df.to_csv(f'./Data Analysis/Best Model Predict Set/{file_name}_predictions.csv', index=False)

def doChekCodeQuality(secondary_terminal):
    
    secondary_terminal.config(state=tk.NORMAL)
    secondary_terminal.insert(tk.END, "Checking Code Quality...\n")
        
    code_quality = CheckCodeQuality.run_flake8() 
    
    secondary_terminal.insert(tk.END, code_quality)
    secondary_terminal.insert(tk.END, "\n")
    secondary_terminal.insert(tk.END, "Finished checking code quality.")
    secondary_terminal.see(tk.END)
    secondary_terminal.config(state=tk.DISABLED)

# Open new window with code quality check
def button2_action():
    secondary_window = tk.Toplevel(root)
    secondary_window.title("Code Quality Check")
    secondary_window.geometry("600x500")

    label = tk.Label(secondary_window, font=("Arial", 16), text="Checking Code Quality...")
    label.place(relx=0.5, rely=0.05, anchor="center") # Centered at the top of the window


    secondary_terminal = tk.Text(secondary_window, bg="black", fg="white", font=("Courier", 10), wrap="word")
    secondary_terminal.place(relx=0.03, rely=0.15, relwidth=0.9, relheight=0.8)  # Fills most of the window

    secondary_window.after(500, lambda: doChekCodeQuality(secondary_terminal))
    

# Create the main window
root = tk.Tk()
root.title("Data Management and Data Analysis")
# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set window dimensions to a percentage of the screen size 
window_width = int(screen_width * 0.8)
window_height = int(screen_height * 0.8)

# Center the window on the screen
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

start = False
monitor_thread = None 
monitor_active = False  


button1 = tk.Button(root, text="Start Data Management", command=button1_action, width=10, height=2, font=("Helvetica", 12))
button1.place(relx=0.01, rely=0.02, relwidth=0.16, relheight=0.1)

button2 = tk.Button(root, text="Check Code Quality", command=button2_action, width=10, height=2, font=("Helvetica", 12))
button2.place(relx=0.63, rely=0.02, relwidth=0.35, relheight=0.1)

button3 = tk.Button(root, text="Start Data Analysis", command=button3_action, width=10, height=2, font=("Helvetica", 12))
button3.place(relx=0.23, rely=0.02, relwidth=0.16, relheight=0.1)

button4 = tk.Button(root, text="Start Prediction", command=button4_action, width=10, height=2, font=("Helvetica", 12))
button4.place(relx=0.45, rely=0.02, relwidth=0.16, relheight=0.1)



# Create a Text widget to simulate the terminal output within the main window
terminal_text = tk.Text(root, bg="black", fg="white", font=("Courier", 10), wrap="word")
terminal_text.place(relx=0.01, rely=0.15, relwidth=0.60, relheight=0.83)  

terminal_text2 = tk.Text(root, bg="black", fg="white", font=("Courier", 10), wrap="word")
terminal_text2.place(relx=0.63, rely=0.15, relwidth=0.35, relheight=0.83)  
write_to_terminal("Click the 'Start Data Management' button to start the data management process :)\n")

# Make terminal read-only initially
terminal_text.config(state=tk.DISABLED)
button3.config(state=tk.DISABLED)
button4.config(state=tk.DISABLED)

# Start the main loop
root.mainloop()





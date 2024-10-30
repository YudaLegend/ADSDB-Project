
import pandas as pd
import json
import duckdb
import os


##################################### Steam Current Player Dataset #####################################
def saveIntoDBCurrentPlayer(con, json_file_path, version):
  # Read the JSON file
  with open(json_file_path, 'r') as file:
      data = json.load(file)

  # Extract the data into a list of dictionaries
  formatted_data = []
  for game_id, game_info in data.items():
      data_info = {}
      game_info_response = game_info['response']
      data_info['app_id'] = game_id
      for k, v in game_info_response.items():
        data_info[k] = v
      formatted_data.append(data_info)
  # Convert the list of dictionaries into a DataFrame
  df = pd.DataFrame(formatted_data)
  # Save the DataFrame to DuckDB (if table already exists, append new data)
  con.execute(f"CREATE TABLE IF NOT EXISTS steam_players_v{version} AS SELECT * FROM df")

def steamCurrentPlayerDataset(con): 
    json_file_path = './Project/Landing Zone/Persistent/steam_api_current/'
    version = 1
    for file_name in os.listdir(json_file_path):
        js_path = json_file_path + file_name
        saveIntoDBCurrentPlayer(con, js_path,version)
        version += 1



##################################### Steam Game Details Dataset #####################################
def saveIntoDBSteamGameInfo(con, json_file_path, version):
  # Read the JSON file
  with open(json_file_path, 'r',encoding='utf-8') as file:
      data = json.load(file)

  # Extract the data into a list of dictionaries
  formatted_data = []
  for app_id, app_info in data.items():
      data_info = {}
      app_data = app_info['data']
      for k, v in app_data.items():
        data_info[k] = v

      formatted_data.append(data_info)

  # Convert the list of dictionaries into a DataFrame
  df = pd.DataFrame(formatted_data)

  df.fillna('', inplace=True)

  # Save the DataFrame to DuckDB (if table already exists, append new data)
  con.execute(f"CREATE TABLE IF NOT EXISTS steam_game_info_v{version} AS SELECT * FROM df")
  # con.close()

def steamGameDetailsDataset(con):
    # Load the JSON data from the file
    json_file_path = './Project/Landing Zone/Persistent/steam_api_details/'
    version = 1
    for file_name in os.listdir(json_file_path):
        js_path = json_file_path + file_name
        saveIntoDBSteamGameInfo(con, js_path,version)
        version += 1



##################################### Steam Spy Dataset #####################################
def saveIntoDBSteamSpy(con, json_file_path, version):

  # Read the JSON file
  with open(json_file_path, 'r',encoding='utf-8') as file:
      data = json.load(file)

  # Extract the data into a list of dictionaries
  formatted_data = []
  for app_id, app_info in data.items():
      data_info = {}
      for k, v in app_info.items():
        data_info[k] = v

      formatted_data.append(data_info)

  # Convert the list of dictionaries into a DataFrame
  df = pd.DataFrame(formatted_data)

  # Save the DataFrame to DuckDB (if table already exists, append new data)
  con.execute(f"CREATE TABLE IF NOT EXISTS steam_spy_v{version} AS SELECT * FROM df")

def steamSpyDataset(con):
    json_file_path = './Project/Landing Zone/Persistent/steam_api_steamspy/'

    version = 1
    for file_name in os.listdir(json_file_path):
        js_path = json_file_path + file_name
        saveIntoDBSteamSpy(con, js_path,version)
        version += 1



##################################### Epic Games Dataset #####################################
def saveIntoDBEpicGames(con, json_file_path, version):

  # Read the JSON file
  with open(json_file_path, 'r') as file:
      data = json.load(file)

  # Extract the data into a list of dictionaries
  formatted_data = []

  for iteration, app_info in data.items():
    app_data = app_info['data']['Catalog']['searchStore']['elements']
    for game in app_data:
      data_info = {}
      for k, v in game.items():
        if isinstance(v, (list, dict)): data_info[k] = json.dumps(v)
        else: data_info[k] = v
      formatted_data.append(data_info)

  df = pd.DataFrame(formatted_data)

  # Save the DataFrame to DuckDB (if table already exists, append new data)
  con.execute(f"CREATE TABLE IF NOT EXISTS epic_games_v{version} AS SELECT * FROM df")
  # con.close()

def epicGamesDataset(con):
    json_file_path = './Project/Landing Zone/Persistent/epic_games_api/'
    
    version = 1
    for file_name in os.listdir(json_file_path):
        js_path = json_file_path + file_name
        # print (js_path)
        saveIntoDBEpicGames(con, js_path,version)
        version += 1


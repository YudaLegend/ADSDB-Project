
import pandas as pd
import json
import os


def open_file(file_path):
  try:
    with open(file_path, 'r', encoding='utf-8') as file:
      data = json.load(file)
  except Exception as e:
     print(f"Error to open the file {file_path}")
     return False
  return data


##################################### Steam Current Player Dataset #####################################
def saveIntoDBCurrentPlayer(con, json_file_path, version):
  # Read the JSON file
  data = open_file(json_file_path)

  if data == False: return False
  # Extract the data into a list of dictionaries
  formatted_data = []
  for game_id, game_info in data.items():
      data_info = {}
      game_info_response = game_info['response']
      data_info['app_id'] = game_id
      for k, v in game_info_response.items():
        data_info[k] = v
      formatted_data.append(data_info)

  try:
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(formatted_data)
  except Exception as e:
     print(f"Error with the steam current player dataset for version {version}: {e}")
     return False
  
  # Save the DataFrame to DuckDB (if table already exists, append new data)
  con.execute(f"CREATE TABLE IF NOT EXISTS steam_players_v{version} AS SELECT * FROM df")
  return True

def steamCurrentPlayerDataset(con): 
    json_file_path = './Landing Zone/Persistent/steam_api_current/'
    version = 1
    for file_name in os.listdir(json_file_path):
        js_path = json_file_path + file_name
        inserted_table = saveIntoDBCurrentPlayer(con, js_path,version)
        if not inserted_table: return False
        version += 1
    return True



##################################### Steam Game Details Dataset #####################################
def saveIntoDBSteamGameInfo(con, json_file_path, version):
  # Read the JSON file
  data = open_file(json_file_path)
  if data == False: return False
  # Extract the data into a list of dictionaries
  formatted_data = []
  for app_id, app_info in data.items():
      data_info = {}
      app_data = app_info['data']
      for k, v in app_data.items():
        data_info[k] = v
      formatted_data.append(data_info)

  try:
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(formatted_data)
    df.fillna('', inplace=True)
  except Exception as e:
     print(f"Error with the stan game details dataset for version {version}: {e}")
     return False

  # Save the DataFrame to DuckDB (if table already exists, append new data)
  con.execute(f"CREATE TABLE IF NOT EXISTS steam_game_info_v{version} AS SELECT * FROM df")
  return True

def steamGameDetailsDataset(con):
    # Load the JSON data from the file
    json_file_path = './Landing Zone/Persistent/steam_api_details/'
    version = 1
    for file_name in os.listdir(json_file_path):
        js_path = json_file_path + file_name
        inserted_table = saveIntoDBSteamGameInfo(con, js_path,version)
        if not inserted_table: return False
        version += 1
    return True



##################################### Steam Spy Dataset #####################################
def saveIntoDBSteamSpy(con, json_file_path, version):

  # Read the JSON file
  data = open_file(json_file_path)
  if data == False: return False
  # Extract the data into a list of dictionaries
  formatted_data = []
  for app_id, app_info in data.items():
      data_info = {}
      for k, v in app_info.items():
        data_info[k] = v

      formatted_data.append(data_info)

  try:
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(formatted_data)
  except Exception as e:
     print(f"Error with the steam spy dataset for version {version}: {e}")
     return False

  # Save the DataFrame to DuckDB (if table already exists, append new data)
  con.execute(f"CREATE TABLE IF NOT EXISTS steam_spy_v{version} AS SELECT * FROM df")
  return True

def steamSpyDataset(con):
    json_file_path = './Landing Zone/Persistent/steam_api_steamspy/'

    version = 1
    for file_name in os.listdir(json_file_path):
        js_path = json_file_path + file_name
        inserted_table = saveIntoDBSteamSpy(con, js_path,version)
        if not inserted_table: return False
        version += 1
    return True



##################################### Epic Games Dataset #####################################
def saveIntoDBEpicGames(con, json_file_path, version):
  # Read the JSON file
  data = open_file(json_file_path)
  if data == False: return False
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

  try: 
    df = pd.DataFrame(formatted_data)
  except Exception as e:
     print(f"Error with the epic games dataset for version {version}: {e}")
     return False
  # Save the DataFrame to DuckDB (if table already exists, append new data)
  con.execute(f"CREATE TABLE IF NOT EXISTS epic_games_v{version} AS SELECT * FROM df")
  return True

def epicGamesDataset(con):
    json_file_path = './Landing Zone/Persistent/epic_games_api/'
    
    version = 1
    for file_name in os.listdir(json_file_path):
        js_path = json_file_path + file_name
        # print (js_path)
        inserted_table = saveIntoDBEpicGames(con, js_path,version)
        if not inserted_table: return False
        version += 1
    return True

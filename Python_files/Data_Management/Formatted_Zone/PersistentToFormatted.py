import pandas as pd
import json
import os


def read_json(file_path):
  try:
    with open(file_path, 'r', encoding='utf-8') as file:
      data = json.load(file)
  except Exception as e:
     print(f"Error to open the file {file_path}")
     return False
  return data


# Save the data into the database creating a new table
def saveIntoDB(table_name, version, formatted_data, con):
   # Convert the list of dictionaries into a DataFrame
  df = pd.DataFrame(formatted_data)
  # Fill NaN values with empty strings to avoid errors when creating tables in DuckDB
  df.fillna('', inplace=True)
  # Save the DataFrame to DuckDB (if table already exists, append new data)
  con.execute(f"CREATE TABLE IF NOT EXISTS {table_name}{version} AS SELECT * FROM df")



# Create a table for a specific data source and version, extracting the data from a JSON file with a specific path
def createTable(json_file_path, datasource, version, con):

  # Read the JSON file where contains the data
  data = read_json(json_file_path)
  if data == False: return False

  # Extract the data into a list of dictionaries
  formatted_data = []
  try:
    # According to the datasource, the extraction of data is different.
    if datasource == 'steam_api_current_players_data':
      # Set the table name for the database
      table_name = 'steam_current_players_v'

      # Extract the data for each game according to the format
      for game_id, game_info in data.items():
          # Get all data attributes for a game
          data_info = {}
          game_info_response = game_info['response']
          data_info['app_id'] = game_id
          for k, v in game_info_response.items():
            data_info[k] = v
          formatted_data.append(data_info)

    elif datasource == 'steam_api_app_details_data':
      # Set the table name for the database
      table_name = 'steam_app_details_v'

      # Extract the data for each game according to the format
      for app_id, app_info in data.items():
          # Get all data attributes for a game
          data_info = {}
          app_data = app_info['data']
          for k, v in app_data.items():
            data_info[k] = v
          formatted_data.append(data_info)

    elif datasource == 'steam_api_steamspy_data':
      # Set the table name for the database
      table_name = 'steam_spy_v'

      # Extract the data for each game according to the format
      for app_id, app_info in data.items():
          # Get all data attributes for a game
          data_info = {}
          for k, v in app_info.items():
            data_info[k] = v
          formatted_data.append(data_info)

    else:
      # Set the table name for the database
      table_name = 'epic_games_v'

      for iteration, app_info in data.items():
        app_data = app_info['data']['Catalog']['searchStore']['elements']
        for game in app_data:
          # Get all data attributes for a game
          data_info = {}
          for k, v in game.items():
            if isinstance(v, (list, dict)): data_info[k] = json.dumps(v)
            else: data_info[k] = v

          formatted_data.append(data_info)
  except Exception as e:
     print(f"Error to extract data from {json_file_path}")
     return False
  
  try:
    # Save the data into the database creating a new table with the specified name and version
    saveIntoDB(table_name, version, formatted_data, con)
  except Exception as e:
     print(f"Error to save data into database for {json_file_path}")
     return False
  return True




def formattedDatasets(con):
  persistent_path = './Data Management/Landing Zone/Persistent/'
  # For each file in the persistent data directory, create a table for each data version
  for file_name in os.listdir(persistent_path):
      dir_path = os.path.join(persistent_path, file_name)
      if os.path.isdir(dir_path):  # Check if it's a directory
          version = 1
          # For each data file in the datasoruce file, create a table for that version
          for file in os.listdir(dir_path):
              json_file_path = dir_path+'/'+file
              inserted_table =  createTable(json_file_path, file_name, version, con)
              if not inserted_table: return False
              version += 1

  return True
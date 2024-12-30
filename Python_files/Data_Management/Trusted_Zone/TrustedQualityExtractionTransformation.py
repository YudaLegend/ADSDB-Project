import re
import json
from datetime import datetime
import ast
import pandas as pd
import duckdb

def read_table(con, table_name):
    return con.execute(f"SELECT * FROM {table_name};").df()

def reload_table(con, table_name, df):
    con.execute(f"DROP TABLE IF EXISTS {table_name};")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

def unify_languages_format(x):
    # Extract languages using regex
    languages = re.findall(r'\b[A-Z][a-zA-Z\s-]+\b(?=<|,|$)', x)
    # Clean up extra spaces
    languages = [lang.strip() for lang in languages]
    if 'languages with full audio support' in languages:
        languages.remove('languages with full audio support')
    # Remove any hyphenated words (e.g., "English-American")
    languages = [re.sub(r'-\w+', '', lang) for lang in languages]
    languages = [re.sub(r' - \w+', '', lang) for lang in languages]
    # Remove duplicates and join the result into a comma-separated string
    cleaned_languages = ', '.join(sorted(set(languages)))
    return cleaned_languages


def extract_name(valor):
    try:
        data = json.loads(valor) 
        return data.get('name', '')  
    except json.JSONDecodeError:
        return ''  

def extract_customAttributes_info(valor, name):
    try:
        lista = json.loads(valor)  
        for item in lista:
            if item.get('key') == name:
                return item.get('value')  
        return '' 
    except json.JSONDecodeError:
        return '' 

def extract_categories(valor):
    try:
        lista = json.loads(valor)  
        
        for item in lista:
            if item['path'] == 'games':
                return ('game') 
        return ''  
    except json.JSONDecodeError:
        return ''

def data_formatted_epic(dt):
        if dt!= '':
            try:
                dt_converted = datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d')
                return dt_converted
            except (ValueError, SyntaxError):
                return 'UNKNOWN'
        else: return 'UNKNOWN' 


def data_formatted_steam(date_str):
    date = date_str.get('date', '')
    if date != '':
        try:
            date_converted = pd.to_datetime(date, format='%d %b, %Y').strftime('%Y-%m-%d')
            return date_converted
        except (ValueError, SyntaxError):
            return date
    else: return 'UNKNOWN'

def convert_to_string(value):
    try:
        list = ast.literal_eval(value)
        return ', '.join(list)
    except (ValueError, SyntaxError):
        return 'UNKNOWN'

def epic_games_extraction_transformation(con):
    df = read_table(con, 'epic_games')
    df['seller'] = df['seller'].apply(extract_name)

    df['publisherName'] = df["customAttributes"].apply(lambda x:extract_customAttributes_info(valor=x,name="publisherName"))
    df['developerName'] = df["customAttributes"].apply(lambda x:extract_customAttributes_info(valor=x,name="developerName"))

    df['categories'] = df["categories"].apply(extract_categories)
    df.drop(columns=["customAttributes"], inplace=True)

    df['effectiveDate'] = df["effectiveDate"].apply(data_formatted_epic)
    df['publisherName'] = df['publisherName'].str.replace(r'\[\]|\{\}', 'UNKNOWN', regex=True)
    df['developerName'] = df['developerName'].str.replace(r'\[\]|\{\}', 'UNKNOWN', regex=True)

    df['publisherName'] = df['publisherName'].fillna('UNKNOWN')
    df['developerName'] = df['developerName'].fillna('UNKNOWN')

    df['publisherName'] = df['publisherName'].apply(lambda x: "UNKNOWN" if x == "" else x.upper())
    df['developerName'] = df['developerName'].apply(lambda x: "UNKNOWN" if x == "" else x.upper())

    reload_table(con, 'epic_games', df)


def steam_app_details_extraction_transformation(con):
    df = read_table(con, 'steam_app_details')
    df['developers'] = df['developers'].apply(convert_to_string)
    df['publishers'] = df['publishers'].apply(convert_to_string)

    df['release_date'] = df['release_date'].apply(data_formatted_steam)

    df['recommendations'] = df['recommendations'].apply(lambda x: ast.literal_eval(x).get("total", '') if x != '' else 0)
    df['achievements'] = df['achievements'].apply(lambda x: ast.literal_eval(x).get('total', '') if x != '' else 0)
    df['metacritic_score'] = df['metacritic'].apply(lambda x: ast.literal_eval(x).get('score', '') if x != '' else 0)
    df['dlc'] = df['dlc'].apply(lambda x: len(ast.literal_eval(x)) if x != '' else 0)


    windows_platform = []
    linux_platform = []
    mac_platform = []
    for i in df['platforms']:
        if i == '':
            windows_platform.append(False)
            linux_platform.append(False)
            mac_platform.append(False)
        else: 
            if isinstance(i, dict): platform = i
            else: platform = ast.literal_eval(i)
            windows_platform.append(platform.get('windows', False))
            linux_platform.append(platform.get('linux', False))
            mac_platform.append(platform.get('mac', False))

    df['windows_platform'] = windows_platform
    df['linux_platform'] = linux_platform
    df['mac_platform'] = mac_platform

    df = df.drop(columns=["metacritic","platforms"])

    df['supported_languages'] = df['supported_languages'].apply(unify_languages_format)

    df['developers'] = df['developers'].apply(lambda x: x.upper())
    df['publishers'] = df['publishers'].apply(lambda x: x.upper())
    df['categories'] = df['categories'].apply(lambda x: "UNKNOWN" if x == "" else x.upper())
    df['genres'] = df['genres'].apply(lambda x: "UNKNOWN" if x == "" else x.upper())
    df['supported_languages'] = df['supported_languages'].apply(lambda x: "UNKNOWN" if x == "" else x.upper())

    reload_table(con, 'steam_app_details', df)


def steam_spy_extraction_transformation(con):
    df = read_table(con, 'steam_spy')

    df['languages'] = df['languages'].apply(unify_languages_format)
    df['developer'] = df['developer'].fillna('')
    df['publisher'] = df['publisher'].fillna('')
    df['languages'] = df['languages'].fillna('')
    df['genre'] = df['genre'].fillna('')

    df['developer'] = df['developer'].apply(lambda x: 'UNKNOWN' if x == '' else x.upper())
    df['publisher'] = df['publisher'].apply(lambda x: 'UNKNOWN' if x == '' else x.upper())
    df['languages'] = df['languages'].apply(lambda x: 'UNKNOWN' if x == '' else x.upper())
    df['genre'] = df['genre'].apply(lambda x: 'UNKNOWN' if x == '' else x.upper())

    df["owners"] = df["owners"].apply(lambda x: sum(map(lambda y: float(y.replace(",", "")), x.split(".."))) / 2)
    df['price'] = df['price'].replace('', 0)
    df['initialprice'] = df['initialprice'].replace('', 0)

    

    reload_table(con, 'steam_spy', df)


def data_extraction_transformation(con):
    try:
        epic_games_extraction_transformation(con)
    except Exception as e:
        print(f"An error occurred while transforming epic_games: {e}")
        return False
    
    try:
        steam_app_details_extraction_transformation(con)
    except Exception as e:
        print(f"An error occurred while transforming steam_app_details: {e}")
        return False
    
    try:
        steam_spy_extraction_transformation(con)
    except Exception as e:
        print(f"An error occurred while transforming steam_spy: {e}")
        return False
    return True
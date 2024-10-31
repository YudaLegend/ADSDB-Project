
import json
import pandas as pd
from currency_converter import CurrencyConverter
from datetime import datetime
import re
from deep_translator import GoogleTranslator
import numpy as np
import ast
import json

# Connect to API of currency converter and translator
c = CurrencyConverter()
translator = GoogleTranslator(source='auto', target='en')


################################## Epic Games Table Functions ##################################
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

def convert_price(valor,name):
    try:
        price_dict = json.loads(valor)  
        if 'totalPrice' in price_dict:
            totalPrice = price_dict['totalPrice']
            if name in totalPrice:
                currencyCode = totalPrice['currencyCode']
                price_converted = c.convert(int(totalPrice[name]), currencyCode, 'EUR')/100
                return round(price_converted,2) 
        return ''
    except Exception as e:
        print(e)
        return ''
    
def data_formatted(dt):
    if dt!= '':
        try:
            dt_converted = datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d')
            return dt_converted
        except (ValueError, SyntaxError):
            return ''
    else: return '' 

################################## Epic Games Table Quality ##################################
def epicGamesQuality(con):
    try:
        # Take the Epic Games table
        df = con.execute("SELECT * FROM  epic_games;").df()
        # Remove duplicate entries
        df = df.drop_duplicates(subset=['id'])
        # Remove useless columns
        df.drop(columns=["namespace", "id", "url", "urlSlug", "keyImages", "items", "tags","productSlug"], inplace=True)
        
        # Extract the useful informations about the game from json format
        df['seller'] = df['seller'].apply(extract_name)
        df['publisherName'] = df["customAttributes"].apply(lambda x:extract_customAttributes_info(valor=x,name="publisherName"))
        df['developerName'] = df["customAttributes"].apply(lambda x:extract_customAttributes_info(valor=x,name="developerName"))
        df['categories'] = df["categories"].apply(extract_categories)
        
        # Extract and convert prices to EUR
        df["originalPrice"] = df["price"].apply(lambda x:convert_price(valor=x,name="originalPrice"))
        df["discountPrice"] = df["price"].apply(lambda x:convert_price(valor=x,name="discountPrice"))

        # Drop the origin columns where extracted the informations
        df.drop(columns=["customAttributes", "price", "categories"], inplace=True)

        # Convert dates to YYYY-MM-DD format
        df['effectiveDate'] = df["effectiveDate"].apply(data_formatted)

        # Replace empty values with 'UNKNOWN' and convert to uppercase
        df['publisherName'] = df['publisherName'].str.replace(r'\[\]|\{\}', 'UNKNOWN', regex=True)
        df['developerName'] = df['developerName'].str.replace(r'\[\]|\{\}', 'UNKNOWN', regex=True)

        df['publisherName'] = df['publisherName'].fillna('UNKNOWN')
        df['developerName'] = df['developerName'].fillna('UNKNOWN')

        df['publisherName'] = df['publisherName'].apply(lambda x: "UNKNOWN" if x == "" else x.upper())
        df['developerName'] = df['developerName'].apply(lambda x: "UNKNOWN" if x == "" else x.upper())

        # Replace the table with all modifications
        con.execute("DROP TABLE epic_games;")
        con.execute("CREATE TABLE IF NOT EXISTS epic_games AS SELECT * FROM df")
    except Exception as e:
        print(f"An error occurred in epicGamesQuality function: {e}")
        return False
    return True


################################## Steam Current Players Table Quality ##################################
def steamCurrentPlayersQuality(con):
    try:
        # Take the Steam Current Players table
        df = con.execute("SELECT * FROM steam_players;").df()
        # Remove duplicate entries
        df.drop_duplicates(subset=['app_id'], inplace=True)
        df['app_id'] = df['app_id'].apply(lambda x: int(x))
        df = df.drop(columns=["result"])

        # Replace the table with all modifications
        con.execute("DROP TABLE steam_players;")
        con.execute("CREATE TABLE IF NOT EXISTS steam_players AS SELECT * FROM df")
    except Exception as e:
        print(f"An error occurred in steamCurrentPlayersQuality function: {e}")
        return False
    return True


################################## Steam Spy Functions ##################################
def unify_languages_format(x):
    languages = re.findall(r'\b[A-Z][a-zA-Z\s-]+\b(?=<|,|$)', x)
    # Remove any extra whitespace
    languages = [lang.strip() for lang in languages]
    return ', '.join(languages)

# Translate the languages to English using Google Translator and cache the results for no translate another time the same word
translation_languages_cache = {}
def translate_languages(x):
    languages = x.split(', ')
    translated_languages = []
    for language in languages:
        if language not in translation_languages_cache:
            translated = translator.translate(language)
            translation_languages_cache[language] = translated
            translated_languages.append(translated)
        else:
            translated = translation_languages_cache[language]
            translated_languages.append(translated)

    translated_languages = ', '.join(translated_languages)
    return translated_languages


# Translate the genres to English using Google Translator and cache the results for no translate another time the same word
translation_genres_cache_spy = {}
def translate_genres_spy(x):
    genres = x.split(', ')
    translated_genres = []
    for genre in genres:
        if genre not in translation_genres_cache_spy:
            translated = translator.translate(genre)
            translation_genres_cache_spy[genre] = translated
            translated_genres.append(translated)
        else:
            translated = translation_genres_cache_spy[genre]
            translated_genres.append(translated)

    translated_genres = ', '.join(translated_genres)
    return translated_genres

################################## Steam Spy Table Quality ##################################
def steamSpyQuality(con):
    try:
        # Take the Steam Spy table
        df = con.execute("SELECT * FROM steam_spy;").df()
        # remove duplicate
        df.drop_duplicates(subset=['appid'], inplace=True)
        # remove useless columns
        df.drop(columns=["score_rank","tags"], inplace=True)

        # Fill NaN values with empty string
        df['developer'] = df['developer'].fillna('')
        df['publisher'] = df['publisher'].fillna('')
        df['languages'] = df['languages'].fillna('')
        df['genre'] = df['genre'].fillna('')

        # Fill empty values with 'UNKNOWN' and convert to uppercase
        df['developer'] = df['developer'].apply(lambda x: 'UNKNOWN' if x == '' else x.upper())
        df['publisher'] = df['publisher'].apply(lambda x: 'UNKNOWN' if x == '' else x.upper())
        df['languages'] = df['languages'].apply(lambda x: 'UNKNOWN' if x == '' else x.upper())
        df['genre'] = df['genre'].apply(lambda x: 'UNKNOWN' if x == '' else x.upper())

        # Calculate the average owners for each game
        df["owners"] = df["owners"].apply(lambda x: sum(map(lambda y: float(y.replace(",", "")), x.split(".."))) / 2)

        # Fill NaN values with 0 for free games
        df['price'] = df['price'].fillna(0)
        df['initialprice'] = df['initialprice'].fillna(0)

        # Convert prices to EUR
        df['price'] = df['price'].apply(lambda x: round (c.convert((float(x) / 100), 'USD', 'EUR'),2) )
        df['initialprice'] = df['initialprice'].apply(lambda x: round (c.convert((float(x) / 100), 'USD', 'EUR'),2) )
        
        # Translate the languages to English 
        df['languages'] = df['languages'].apply(translate_languages)
        
        # Translate the genres to English 
        df['genre'] = df["genre"].apply(translate_genres_spy)

        # Replace NaN values to 'UNKNOWN'
        df['name'].replace(np.nan, 'UNKNOWN', inplace=True)
        df['discount'].replace(np.nan, 0, inplace=True)

        # Reload the table with all modifications
        con.execute("DROP TABLE steam_spy;")
        con.execute("CREATE TABLE IF NOT EXISTS steam_spy AS SELECT * FROM df")
    except Exception as e:
        print(f"An error occurred in steamSpyQuality function: {e}")
        return False
    return True

################################## Steam Games Details Functions ##################################
def convert_to_string(value):
    try:
        list = ast.literal_eval(value)
        return ', '.join(list)
    except (ValueError, SyntaxError):
        return 'UNKNOWN'
    
def data_formatted_detail(date_str):
    date = date_str.get('date', '')
    if date != '':
        try:
            date_converted = pd.to_datetime(date, format='%d %b, %Y').strftime('%Y-%m-%d')
            return date_converted
        except (ValueError, SyntaxError):
            return date
    else: return 'UNKNOWN'

# Convert prices to EUR
def currencyConvert(df):
    initial_price = []
    final_price = []

    ## Put all currencies that don't exist in the currency_converter library to EUR conversion
    currencyToEUR = {'COP': 0.00022, 'CLP': 0.00098, 'VND': 0.000037, 'PEN': 0.25, 'AED': 0.25, 'UAH': 0.022, 'SAR': 0.25, 'KZT': 0.0019}
    for i in df['price_overview']:
        if i != '':
            try:
                price_dict = ast.literal_eval(i)
                
                from_currency = price_dict['currency']
                initial = price_dict['initial']
                discount = price_dict['final']
                if from_currency in currencyToEUR:
                    initial_price.append(int(initial)/100*currencyToEUR[from_currency])
                    final_price.append(int(discount)/100*currencyToEUR[from_currency])
                else:
                    initial_price.append(c.convert(int(initial), from_currency, 'EUR')/100)
                    final_price.append(c.convert(int(discount), from_currency, 'EUR')/100)
            except Exception as e:
                print(e)
                initial_price.append(0)
                final_price.append(0)
        else:
            initial_price.append(0)
            final_price.append(0)

    return initial_price, final_price


# Translate the text using Google Translator and cache the results for no translate another time the same word
translation_cache = {}
def translate_with_cache(text):
    if text not in translation_cache:
        translation_cache[text] = translator.translate(text)
    return translation_cache[text]

def translate_text(text_dict):
    allText = []
    for i in text_dict:
        text = i.get('description', '')
        if text != '':
            allText.append(translate_with_cache(text))
    try:
        translated_text = (', '.join(allText))
    except Exception as e:
        print(f"An error occurred: {e}")
    return translated_text

def getTranslated(genre, category):
    genre_translated = ''
    category_translated = ''
    if genre != '':
        if not isinstance(genre, dict): genre = ast.literal_eval(genre) 
        try:
            genre_translated = translate_text(genre)
        except Exception as e:
            print(f"An error occurred: {e}")
    if category != '':
        if not isinstance(category, dict): category = ast.literal_eval(category) 
        try:
            category_translated = translate_text(category)
        except Exception as e:
            print(f"An error occurred: {e}")
    return genre_translated, category_translated

def unify_languages_format_detail(x):
    languages = re.findall(r'\b[A-Z][a-zA-Z\s-]+\b(?=<|,|$)', x)
    # Remove any extra whitespace
    languages = [lang.strip() for lang in languages]
    
    if 'languages with full audio support' in languages:
        languages.remove('languages with full audio support')

    return ', '.join(languages)
    
################################## Steam Games Details Table Quality ##################################
def steamGameDetailsQuality(con):
    try:
        # Take the Steam Games Details table
        df = con.execute("SELECT * FROM steam_game_info;").df()
        # Remove duplicate entries
        df.drop_duplicates(subset=['steam_appid'], inplace=True)
        # Remove useless columns
        df.drop(columns=['controller_support', 'header_image', 'capsule_image', 'capsule_imagev5', 'website', 'legal_notice', 'pc_requirements',
                        'mac_requirements', 'linux_requirements', 'screenshots', 'movies', 'ext_user_account_notice', 'drm_notice', 
                        'support_info', 'background', 'background_raw', 'reviews', 'content_descriptors', 'ratings', 
                        'demos', 'packages','package_groups',"fullgame"], inplace=True)
        
        # Convert json format to string
        df['developers'] = df['developers'].apply(convert_to_string)
        df['publishers'] = df['publishers'].apply(convert_to_string)
        
        # Convert release date to YYYY-MM-DD format
        df['release_date'] = df['release_date'].apply(data_formatted_detail)

        # Extract the useful informations from json format and fill NaN values with 0
        df['recommendations'] = df['recommendations'].apply(lambda x: ast.literal_eval(x).get("total", '') if x != '' else 0)
        df['achievements'] = df['achievements'].apply(lambda x: ast.literal_eval(x).get('total', '') if x != '' else 0)
        df['metacritic_score'] = df['metacritic'].apply(lambda x: ast.literal_eval(x).get('score', '') if x != '' else 0)

        # Extract the number of dlc of the game
        df['dlc'] = df['dlc'].apply(lambda x: ', '.join(map(str, ast.literal_eval(x))) if x != '' else '')
        df['dlc'] = df['dlc'].apply(lambda x: len(x))

        # Convert prices to EUR
        initial_price, final_price = currencyConvert(df)
        df['initial_price'] = initial_price
        df['final_price'] = final_price

        # Extract the platforms of the game
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

        # Drop the origin columns where extracted the informations
        df = df.drop(columns=["metacritic","platforms","price_overview"])

        # Translate the text using Google Translator
        genres_translated = []
        categories_translated = []
        for index, row in df[['categories', 'genres']].iterrows():
            genre = row['genres']
            category = row['categories']
            genre_translated, category_translated = getTranslated(genre, category)

            genres_translated.append(genre_translated)
            categories_translated.append(category_translated)
        ## This translation may take a long of time, because it using an API to translate
        df['genres'] = genres_translated
        df['categories'] = categories_translated

        # Unify the format of the languages
        df['supported_languages'] = df['supported_languages'].apply(unify_languages_format_detail)

        # Remove the country of languages
        df['supported_languages'] = df['supported_languages'].apply(lambda x: re.sub(r' - \w+', '', x))
        df['supported_languages'] = df['supported_languages'].apply(lambda x: ', '.join(list(set(x.split(', ')))))
        df['supported_languages'] = df['supported_languages'].apply(translate_languages)

        # Replace NaN values to 'UNKNOWN' and convert to uppercase
        df['developers'] = df['developers'].apply(lambda x: x.upper())
        df['publishers'] = df['publishers'].apply(lambda x: x.upper())
        df['categories'] = df['categories'].apply(lambda x: "UNKNOWN" if x == "" else x.upper())
        df['genres'] = df['genres'].apply(lambda x: "UNKNOWN" if x == "" else x.upper())
        df['supported_languages'] = df['supported_languages'].apply(lambda x: "UNKNOWN" if x == "" else x.upper())

        # Replace empty values to 'UNKNOWN'
        df['name'].replace('', 'UNKNOWN', inplace=True)
        df['detailed_description'].replace('', 'UNKNOWN', inplace=True)
        df['about_the_game'].replace('', 'UNKNOWN', inplace=True)
        df['short_description'].replace('', 'UNKNOWN', inplace=True)

        # Reload the table with all modifications
        con.execute("DROP TABLE steam_game_info;")
        con.execute("CREATE TABLE IF NOT EXISTS steam_game_info AS SELECT * FROM df")
    
    except Exception as e:
        print(f"An error occurred in steamGameDetailsQuality function: {e}")
        return False
    return True

    
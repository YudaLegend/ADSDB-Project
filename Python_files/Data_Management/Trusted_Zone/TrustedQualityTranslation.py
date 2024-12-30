from deep_translator import GoogleTranslator
import ast
import duckdb

def read_table(con, table_name):
    return con.execute(f"SELECT * FROM {table_name};").df()

def reload_table(con, table_name, df):
    con.execute(f"DROP TABLE IF EXISTS {table_name};")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

translation_cache = {}
translator = GoogleTranslator(source='auto', target='en')

def translate_with_cache(text):
    if text not in translation_cache:
        translation_cache[text] = translator.translate(text)
    return translation_cache[text]

def translate_items(x):
    items = x.split(', ')
    translated_items = []
    
    for item in items:
        translated_items.append(translate_with_cache(item).upper())
    translated_items = ', '.join(translated_items)
    return translated_items

def translate_text(text_dict):
    allText = []
    for i in text_dict:
        text = i.get('DESCRIPTION', '')
        if text != '':
            allText.append(translate_with_cache(text).upper())
    try:
        translated_text = ', '.join(allText)
    except Exception as e:
        print(f"An error occurred: {e}")
    return translated_text

def getTranslated(genre, category):
    genre_translated = ''
    category_translated = ''

    if genre == 'UNKNOWN':
        genre_translated = 'UNKNOWN'
    elif genre != '':
        if not isinstance(genre, dict): genre = ast.literal_eval(genre) 
        try:
            genre_translated = translate_text(genre)
        except Exception as e:
            print(f"An error occurred: {e}")

    if category == 'UNKNOWN':
        category_translated = 'UNKNOWN'        
    elif category != '':
        if not isinstance(category, dict): category = ast.literal_eval(category) 
        try:
            category_translated = translate_text(category)
        except Exception as e:
            print(f"An error occurred: {e}")
    return genre_translated, category_translated

def delete_languages_support(x):
    languages = x.split(', ')
    # Remove the country of the languages, only a few of them have this problem
    languages = [lang.replace('PORTUGUESE FROM PORTUGAL', 'PORTUGUESE') for lang in languages]
    languages = [lang.replace('PORTUGAL', 'PORTUGUESE') for lang in languages]
    languages = [lang.replace('IN ENGLISH', 'ENGLISH') for lang in languages]
    languages = [lang.replace('SPANISH AMERICA', 'SPANISH') for lang in languages]
    languages = [lang.replace('ITALY', 'ITALIAN') for lang in languages]
    if 'LANGUAGES WITH FULL AUDIO SUPPORT' in languages:
        languages.remove('LANGUAGES WITH FULL AUDIO SUPPORT')
    if 'FROM' in languages:
        languages.remove('FROM')
    return ', '.join(set(sorted(languages)))


def steam_app_details_translation(con):
    df = read_table(con, 'steam_app_details')
    

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

    df['supported_languages'] = df['supported_languages'].apply(translate_items)
    df['supported_languages'] = df['supported_languages'].apply(delete_languages_support)


    df['name'].replace('', 'UNKNOWN', inplace=True)
    df['detailed_description'].replace('', 'UNKNOWN', inplace=True)
    df['about_the_game'].replace('', 'UNKNOWN', inplace=True)
    df['short_description'].replace('', 'UNKNOWN', inplace=True)

    reload_table(con, 'steam_app_details', df)


def steam_spy_translation(con):
    df = read_table(con, 'steam_spy')
    df['languages'] = df['languages'].apply(translate_items)
    df['languages'] = df['languages'].apply(delete_languages_support)
    # Apply the function to each genre in the DataFrame
    df['genre'] = df["genre"].apply(translate_items)

    df['name'].replace('', 'UNKNOWN', inplace=True)
    df['discount'].replace('', 0, inplace=True)
    
    reload_table(con, 'steam_spy', df)



def data_translation(con):
    try:
        steam_app_details_translation(con)
    except Exception as e:
        print(f"An error occurred in steam app details translation: {e}")
        return False

    try:
        steam_spy_translation(con)
    except Exception as e:
        print(f"An error occurred in steam spy translation: {e}")
        return False
    return True
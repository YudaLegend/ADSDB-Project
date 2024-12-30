# This file contains the code to convert categorical values to numerical values based on their proportion in the dataset
import numpy as np
import pandas as pd
import duckdb

# Get the proportion of each categorical value in the dataset
def getProportionOfCategorical(col):
    col = col.replace('UNKNOWN', np.nan)
    flat_col_data = col.dropna().apply(lambda x: x.split(', ')).explode()
    freq_dist = flat_col_data.value_counts()
    freq_dist = freq_dist/sum(freq_dist)
    return dict(freq_dist)

# Convert categorical values to numerical values based on their proportion in the dataset
def converToNumerical(items, proportion):
    
    items_list = items.split(', ')
    total_proportion = 0
    for item in items_list:
        if item == 'UNKNOWN': return 0
        total_proportion += proportion[item]
    return total_proportion

# Convert categorical values to numerical values based on their proportion in the dataset
def featureGenerationConvertToNumerical(con,con1):
    try:
        df = con.execute("SELECT * FROM sandbox_steam_games_kpi;").df()
        
        proportionLanguages = getProportionOfCategorical(df['languages'])
        proportionGenres = getProportionOfCategorical(df['genres'])
        proportionCategories = getProportionOfCategorical(df['categories'])


        df['languages'] = df['languages'].apply(lambda x: converToNumerical(x, proportionLanguages))
        df['genres'] = df['genres'].apply(lambda x: converToNumerical(x, proportionGenres))
        df['categories'] = df['categories'].apply(lambda x: converToNumerical(x, proportionCategories))
        con1.execute("CREATE TABLE IF NOT EXISTS feature_steam_games AS SELECT * FROM df")
    except Exception as e:
        print("An error occurred while converting categorical to numerical: ", e)
        return False
    return True


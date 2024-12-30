
import numpy as np
import pandas as pd
import duckdb

def categorize_numerical(value, quantiles):
    if value <= quantiles[0]:
        return "low"
    elif value <= quantiles[1]:
        return "medium"
    else:
        return "high"

def featureGenerationConvertToCategorical(con):
    try:
        df = con.execute("SELECT * FROM feature_steam_games;").df()
        quantiles = df['recommendation_ratio'].quantile([0.33, 0.66]).values
        df['recommendation_ratio'] = df['recommendation_ratio'].apply(lambda x: categorize_numerical(x, quantiles))

        quantiles = df['game_popularity'].quantile([0.33, 0.66]).values
        df['game_popularity'] = df['game_popularity'].apply(lambda x: categorize_numerical(x, quantiles))

        con.execute("DROP TABLE feature_steam_games;")
        con.execute("CREATE TABLE IF NOT EXISTS feature_steam_games AS SELECT * FROM df")
    except Exception as e:
        print("An error occurred while converting numerical to categorical: ", e)
        return False
    return True



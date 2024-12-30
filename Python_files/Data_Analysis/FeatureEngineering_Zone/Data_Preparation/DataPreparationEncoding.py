# This file contains the code to encode the categorical variables in the dataset using Label Encoding technique.
import duckdb
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

labelEncoder = LabelEncoder()

# This function encodes the categorical variables in the dataset using the Label Encoding technique.
def dataPreparationEncoding(con):
    try:
        df = con.execute('SELECT * FROM feature_steam_games').df()
        df['game_popularity'] = labelEncoder.fit_transform(df["game_popularity"])
        df['recommendation_ratio'] = labelEncoder.fit_transform(df["recommendation_ratio"])
        con.execute("DROP TABLE feature_steam_games;")
        con.execute("CREATE TABLE IF NOT EXISTS feature_steam_games AS SELECT * FROM df")
    except Exception as e:
        print("An error occurred while encoding the categorical variables: ", e)
        return False
    return True




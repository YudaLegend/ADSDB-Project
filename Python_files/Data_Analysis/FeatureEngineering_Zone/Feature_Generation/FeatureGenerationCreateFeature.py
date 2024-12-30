# This file contains the code to create a new feature in the dataset. The new feature is the price discount of the games. The price discount is calculated as the difference between the initial price and the final price divided by the initial price. The new feature is added to the dataset and the initial and final price columns are dropped. The new dataset is saved in the database as a new table called feature_steam_games.
import numpy as np
import pandas as pd
import duckdb

# Create a new feature in the dataset
def featureGenerationCreateFeature(con):
    try:
        df = con.execute("SELECT * FROM feature_steam_games;").df()
        price_discount = (df['initial_price'] - df['final_price'])/ df['initial_price']
        price_discount.replace(np.nan, 0, inplace=True)
        df['price_discount'] = price_discount
        df.drop(columns=['initial_price', 'final_price'], inplace=True)
        con.execute("DROP TABLE feature_steam_games;")
        con.execute("CREATE TABLE IF NOT EXISTS feature_steam_games AS SELECT * FROM df")
    except Exception as e:
        print("An error occurred while creating the feature: ", e)
        return False
    return True


# This script is used to prepare the data for the model building process. The script handles missing data and outliers in the dataset. The missing data is imputed using the IterativeImputer class from the sklearn.impute module. The outliers are detected using the IQR method and replaced with NaN values. The prepared data is then saved in the feature_steam_games table for further processing.
import duckdb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np

# Change all outliers to NaN
def outliers_data(df, columns):
    for col in columns:
        col_data = df[col]
        # IQR and Outlier Detection
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = col_data[(col_data < Q1 - 3 * IQR) | (col_data > Q3 + 3 * IQR)]
        # print(outliers)
        df.loc[outliers.index, col] = np.nan

# Handle missing data and outliers in the dataset
def dataPreparationOutlierMissingData(con):
    try:
        df = con.execute('SELECT * FROM feature_steam_games').df()
        
        outliers_data(df, ['price_discount', 'average_playtime', 'median_playtime', 'game_active_players_2days'])
        
        df['game_satisfaction'].replace(np.nan, 0, inplace=True)

        # Use the `IterativeImputer` class from the `sklearn.impute` module to impute missing values in the dataset (no target variable). The `IterativeImputer` class is an imputation method that iteratively estimates the missing values using a regression model. The `fit_transform` method of the `IterativeImputer` class is used to impute the missing values in the dataset.

        imputer = IterativeImputer(max_iter=100, random_state=42)
        df_impute_col = df[['languages', 'genres', 'categories', 'game_active_players_2days', 'average_playtime', 'median_playtime', 'price_discount']]
        imputed_df = pd.DataFrame(imputer.fit_transform(df_impute_col), columns=df_impute_col.columns)

        # Assign the imputed data to the corresponding variables.

        df[['game_active_players_2days','average_playtime','median_playtime','price_discount']] = imputed_df[['game_active_players_2days','average_playtime','median_playtime','price_discount']]

        # Reload the prepared data in `feature_steam_games` table, prepared for the model splitting (train and test sets) and model building.

        con.execute("DROP TABLE feature_steam_games;")
        con.execute("CREATE TABLE IF NOT EXISTS feature_steam_games AS SELECT * FROM df")
    except Exception as e:
        print("An error occurred while handling missing data: ", e)
        return False
    return True

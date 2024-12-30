import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Python_files/Data_Analysis/FeatureEngineering_Zone/Data_Preparation')))

import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from DataPreparationOutlierMissingData import dataPreparationOutlierMissingData   # type: ignore

class TestDataPreparationOutlierMissingData(unittest.TestCase):
    def setUp(self):
        # Set up mock for the database connection (duckdb)
        self.mock_con = MagicMock()
        
        # Sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'price_discount': [5, 10, 200, 15, -10],  # outlier in 200 and -10
            'average_playtime': [50, 20, 150, 60, 30],  # outlier in 150
            'median_playtime': [45, 19, 140, 59, 28],  # outlier in 140
            'game_active_players_2days': [1000, 200, 5000, 1200, 300],  # outlier in 5000
            'game_satisfaction': [1, 2, np.nan, 4, 5],  # missing value in third row
            'languages': [1, 2, 3, 4, 5],
            'genres': [1, 2, 3, 4, 5],
            'categories': [1, 2, 3, 4, 5]
        })

        # Mock the query to return the sample data
        self.mock_con.execute.return_value.df.return_value = self.sample_data

    def test_data_preparation_outlier_missing_data(self):
        # Call the function with the mocked connection
        result = dataPreparationOutlierMissingData(self.mock_con)
        
        # Check if the function returned True (no error)
        self.assertTrue(result)

        # Check if the mock's execute method was called to drop the old table
        self.mock_con.execute.assert_any_call("DROP TABLE feature_steam_games;")
        
        # Check if the mock's execute method was called to create a new table
        self.mock_con.execute.assert_any_call("CREATE TABLE IF NOT EXISTS feature_steam_games AS SELECT * FROM df")

        # Check that the imputation of missing data was done correctly by comparing the imputed columns
        # The columns should now have no missing values and have imputed values for outliers
        imputed_data = self.mock_con.execute.return_value.df.return_value[['game_active_players_2days', 'average_playtime', 'median_playtime', 'price_discount']]
        
        imputer = IterativeImputer(max_iter=100, random_state=42)
        df_impute_col = self.sample_data[['languages', 'genres', 'categories', 'game_active_players_2days', 'average_playtime', 'median_playtime', 'price_discount']]
        imputed_df = pd.DataFrame(imputer.fit_transform(df_impute_col), columns=df_impute_col.columns)
        
        # Ensure the imputed values are correct
        pd.testing.assert_frame_equal(imputed_data, imputed_df[['game_active_players_2days', 'average_playtime', 'median_playtime', 'price_discount']], check_dtype=False)

    def test_data_preparation_outlier_missing_data_exception(self):
        # Simulate an error in the process by making the mock raise an exception
        self.mock_con.execute.side_effect = Exception("Mocked database error")

        # Call the function and check for failure
        result = dataPreparationOutlierMissingData(self.mock_con)
        
        # Verify the function returns False when an error occurs
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()

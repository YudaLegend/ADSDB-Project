import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Python_files/Data_Analysis/FeatureEngineering_Zone/Feature_Generation')))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from FeatureGenerationConvertToCategorical import featureGenerationConvertToCategorical # type: ignore


class TestFeatureGenerationConvertToCategorical(unittest.TestCase):
    def setUp(self):
        # Set up mock for database connection (duckdb)
        self.mock_con = MagicMock()
        
        # Sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'game_popularity': [1, 2, 3, 4, 5],
            'recommendation_ratio': [1, 2, 3, 4, 5]
        })
        
        # Mock the query to return the sample data
        self.mock_con.execute.return_value.df.return_value = self.sample_data

    def test_feature_generation_convert_to_categorical(self):
        # Create an instance of the feature generation class
        result = featureGenerationConvertToCategorical(self.mock_con)
        
        # Check if the function returned True (no error)
        self.assertTrue(result)

        # Check if the mock's execute method was called to drop the old table
        self.mock_con.execute.assert_any_call("DROP TABLE feature_steam_games;")
        
        # Check if the mock's execute method was called to create a new table
        self.mock_con.execute.assert_any_call("CREATE TABLE IF NOT EXISTS feature_steam_games AS SELECT * FROM df")

    def test_feature_generation_convert_to_categorical_exception(self):
        # Simulate an error in the process by making the mock raise an exception
        self.mock_con.execute.side_effect = Exception("Mocked database error")

        # Call the function and check for failure
        result = featureGenerationConvertToCategorical(self.mock_con)
        
        # Verify the function returns False when an error occurs
        self.assertFalse(result)
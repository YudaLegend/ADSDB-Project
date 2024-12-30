import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Python_files/Data_Analysis/FeatureEngineering_Zone/Feature_Generation')))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from FeatureGenerationCreateFeature import featureGenerationCreateFeature # type: ignore


class TestFeatureGenerationCreateFeature(unittest.TestCase):

    def setUp(self):
        # Set up mock for database connection (duckdb)
        self.mock_con = MagicMock()
        
        # Sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'initial_price': [1, 2, 3, 4, 5],
            'final_price': [1, 1, 2, 3, 5]
        })
        
        # Mock the query to return the sample data
        self.mock_con.execute.return_value.df.return_value = self.sample_data

    def test_feature_generation_create_feature(self):
        # Call the function to generate the feature
        result = featureGenerationCreateFeature(self.mock_con)

        # Check if the function returned True (no error)
        self.assertTrue(result)

        # Check if the mock's execute method was called to drop the old table
        self.mock_con.execute.assert_any_call("DROP TABLE feature_steam_games;")
        
        # Check if the mock's execute method was called to create a new table
        self.mock_con.execute.assert_any_call("CREATE TABLE IF NOT EXISTS feature_steam_games AS SELECT * FROM df")

    
    def test_feature_generation_create_feature_exception(self):
        # Mock the execute method to raise an exception
        self.mock_con.execute.side_effect = Exception("An error occurred")
        
        # Call the function to generate the feature
        result = featureGenerationCreateFeature(self.mock_con)
        
        # Check if the function returned False (an error occurred)
        self.assertFalse(result)
        
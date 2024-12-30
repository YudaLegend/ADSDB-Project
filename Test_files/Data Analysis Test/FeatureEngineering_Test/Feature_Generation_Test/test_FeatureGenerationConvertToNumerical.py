import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Python_files/Data_Analysis/FeatureEngineering_Zone/Feature_Generation')))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from FeatureGenerationConvertToNumerical import featureGenerationConvertToNumerical # type: ignore

class TestFeatureGenerationConvertToNumerical(unittest.TestCase):
    def setUp(self):
        # Set up mock for the database connection (duckdb)
        self.mock_con = MagicMock()
        self.mock_con1 = MagicMock()
        
        # Sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'languages': ['English', 'Spanish', 'English', 'UNKNOWN', 'Spanish'],
            'genres': ['Action', 'Adventure', 'Action', 'Action', 'Adventure'],
            'categories': ['Cat1, Cat2', 'Cat3', 'Cat1', 'Cat2', 'Cat3']
        })

        # Mock the query to return the sample data
        self.mock_con.execute.return_value.df.return_value = self.sample_data

    def test_feature_generation_convert_to_numerical(self):
        # Call the function with the mocked connections
        result = featureGenerationConvertToNumerical(self.mock_con, self.mock_con1)
        
        # Check if the function returned True (no error)
        self.assertTrue(result)

        # Check if the 'languages' column was updated correctly
        self.mock_con.execute.assert_called_once_with("SELECT * FROM sandbox_steam_games_kpi;")

        # Check that the apply function was used on columns and the correct value is passed
        self.mock_con1.execute.assert_called_once_with("CREATE TABLE IF NOT EXISTS feature_steam_games AS SELECT * FROM df")

    def test_feature_generation_convert_to_numerical_exception(self):
        # Simulate an error in the process by making the mock raise an exception
        self.mock_con.execute.side_effect = Exception("Mocked database error")

        # Call the function and check for failure
        result = featureGenerationConvertToNumerical(self.mock_con, self.mock_con1)
        
        # Verify the function returns False when an error occurs
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
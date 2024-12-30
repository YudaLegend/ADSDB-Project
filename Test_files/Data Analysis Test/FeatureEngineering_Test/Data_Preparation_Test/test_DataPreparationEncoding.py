import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Python_files/Data_Analysis/FeatureEngineering_Zone/Data_Preparation')))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DataPreparationEncoding import dataPreparationEncoding  # type: ignore

class TestDataPreparationEncoding(unittest.TestCase):
    def setUp(self):
        # Set up mock for database connection (duckdb)
        self.mock_con = MagicMock()
        
        # Sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'game_popularity': ['High', 'Low', 'Low', 'High', 'Low'],
            'recommendation_ratio': ['High', 'High', 'Low', 'Low', 'High']
        })
        
        # Mock the query to return the sample data
        self.mock_con.execute.return_value.df.return_value = self.sample_data

    def test_data_preparation_encoding(self):
        # Call the function with the mocked connection
        result = dataPreparationEncoding(self.mock_con)
        
        # Check if the function returned True (no error)
        self.assertTrue(result)

        # Check if the mock's execute method was called to drop the old table
        self.mock_con.execute.assert_any_call("DROP TABLE feature_steam_games;")
        
        # Check if the mock's execute method was called to create a new table
        self.mock_con.execute.assert_any_call("CREATE TABLE IF NOT EXISTS feature_steam_games AS SELECT * FROM df")
        
        # Verify if the categorical columns were encoded using LabelEncoder
        encoded_data = self.mock_con.execute.return_value.df.return_value
        label_encoder = LabelEncoder()

        # Apply label encoding manually for validation
        self.assertTrue(
            (encoded_data['game_popularity'] == label_encoder.fit_transform(self.sample_data['game_popularity'])).all()
        )
        self.assertTrue(
            (encoded_data['recommendation_ratio'] == label_encoder.fit_transform(self.sample_data['recommendation_ratio'])).all()
        )

    def test_data_preparation_encoding_exception(self):
        # Simulate an error in the encoding process by making the mock raise an exception
        self.mock_con.execute.side_effect = Exception("Mocked database error")

        # Call the function and check for failure
        result = dataPreparationEncoding(self.mock_con)
        
        # Verify the function returns False when an error occurs
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Python_files/Data_Analysis/FeatureEngineering_Zone')))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from GenerationTrainTest import  generationTrainTest # type: ignore


class TestGenerationTrainTest(unittest.TestCase):
    def setUp(self):
        # Set up mock for database connection (duckdb)
        self.mock_con = MagicMock()
        
        # Sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'languages': np.linspace(0.1, 1, 10),
            'genres': np.linspace(0.1, 1, 10),
            'categories': np.linspace(0.1, 1, 10),
            'game_popularity': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            'recommendation_ratio': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            'game_satisfaction': np.linspace(0.1, 1, 10)
        })
        
        # Mock the query to return the sample data
        self.mock_con.execute.return_value.df.return_value = self.sample_data

    def test_generation_train_test(self):
        # Call the function to generate the train and test sets
        result = generationTrainTest(self.mock_con)

        # Check if the function returned True (no error)
        self.assertTrue(result)

         # Check if the mock's execute method was called to create a new tables
        self.mock_con.execute.assert_any_call("CREATE TABLE IF NOT EXISTS train_dataset AS SELECT * FROM train")
        self.mock_con.execute.assert_any_call("CREATE TABLE IF NOT EXISTS test_dataset AS SELECT * FROM test")

    def test_generation_train_test_exception(self):
        # Simulate an error in the process by making the mock raise an exception
        self.mock_con.execute.side_effect = Exception("Mocked database error")

        # Call the function and check for failure
        result = generationTrainTest(self.mock_con)
        
        # Verify the function returns False when an error occurs
        self.assertFalse(result)

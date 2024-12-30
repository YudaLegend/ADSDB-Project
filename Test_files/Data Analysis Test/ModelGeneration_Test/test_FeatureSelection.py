import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Python_files/Data_Analysis/ModelGeneration_Zone')))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from FeatureSelection import varianceThreshold, rRegression, fRegression # type: ignore

class TestFeatureSelection(unittest.TestCase):

    def setUp(self):
        # Mock sample data
        np.random.seed(123)
        self.X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'feature4': np.random.rand(100),
        })
        self.y = np.random.rand(100)

    def test_varianceThreshold(self):
        # Test if variance thresholding returns the correct features
        result, features = varianceThreshold(self.X)
        self.assertTrue(result)  # Check if the result is True
        self.assertIsInstance(features, str)  # Features should be a comma-separated string

    def test_rRegression(self):
        # Test if r-regression works and returns expected output
        result, correlated_vars = rRegression(self.X, self.y)
        self.assertTrue(result)  # Ensure the result is True
        self.assertIsInstance(correlated_vars, str)  # Check if the return type is a string
        # We should expect at least one correlated feature
        self.assertGreater(len(correlated_vars), 0)  

    def test_fRegression(self):
        # Test if f-regression works and returns expected output
        result, significant_vars = fRegression(self.X, self.y)
        self.assertTrue(result)  # Ensure the result is True
        self.assertIsInstance(significant_vars, str)  # Check if the return type is a string

if __name__ == '__main__':
    unittest.main()
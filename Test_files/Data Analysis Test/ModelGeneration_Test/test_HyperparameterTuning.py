import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from Python_files.Data_Analysis.ModelGeneration_Zone.HyperparameterTuning import hyperparameter_tuning

class TestHyperparameterTuning(unittest.TestCase):
    def setUp(self):
        # Mock sample data
        self.X = pd.DataFrame({
            'languages': [0.1, 0.2, 0.3],
            'genres': [0.1, 0.2, 0.3],
            'categories': [0.1, 0.2, 0.3],
            'game_popularity': [0, 0, 1],
            'game_active_players_2days': [1000, 2000, 1500],
            'recommendation_ratio': [1, 0, 0],
            'average_playtime': [10, 15, 20],
            'median_playtime': [5, 7, 10],
            'price_discount': [0.2, 0.1, 0.15],
        })

        self.y = [4.5, 4.7, 4.6]

    @patch("Python_files.DataAnalysis.ModelGeneration_Zone.HyperparameterTuning.HalvingGridSearchCV")
    def test_hyperparameter_tuning(self, mock_hgscv):
        
        # Mock the HalvingGridSearchCV instance and its fit method
        mock_hgscv_instance = MagicMock()
        mock_hgscv_instance.fit.return_value = MagicMock(
            best_params_={
                'hidden_layers': (64, 32, 16),
                'dropout_rate': 0.2,
                'lr': 0.003,
                'wd': 0.0003,
                'batch_size': 128,
            },
            best_score_=0.26
        )
        mock_hgscv.return_value = mock_hgscv_instance

        # Run the function
        success, result = hyperparameter_tuning(self.X, self.y)

        # Assertions
        self.assertTrue(success)
        mock_hgscv.assert_called_once()  # Check if HalvingGridSearchCV was called
        self.assertIsNotNone(result)  # Ensure we got a result
        self.assertIn('hidden_layers', result.best_params_)  # Check if best params are present
        self.assertEqual(result.best_score_, 0.26)  # Verify the mocked best score
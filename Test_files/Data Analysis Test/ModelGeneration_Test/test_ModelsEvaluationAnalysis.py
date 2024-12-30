import os
import sys
import numpy as np
from sklearn.metrics import root_mean_squared_error, r2_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Python_files/Data_Analysis/ModelGeneration_Zone')))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from ModelsEvaluationAnalysis import metric_results, model_predict, frequency_distribution # type: ignore

class TestModelsEvaluationAnalysis(unittest.TestCase):
    def setUp(self):
        # Mock dataset
        self.X_test = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        self.y_test = np.array([3.0, 2.5, 4.0, 5.0, 3.5])
        self.y_pred = np.array([3.1, 2.4, 3.8, 5.2, 3.6])
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = self.y_pred
        self.y_preds = [
            self.y_pred + 0.1,  # Mock predictions for sgdLinearRegressor
            self.y_pred - 0.2,  # Mock predictions for randomForestRegressor
            self.y_pred + 0.05, # Mock predictions for mlpRegressor
            self.y_pred         # Mock predictions for nfnnRegressor
        ]

    def test_metric_results(self):
        # Test r2_score and RMSE calculation

        r2, rmse = metric_results(self.y_test, self.y_pred)
        self.assertAlmostEqual(r2, r2_score(self.y_test, self.y_pred))
        self.assertAlmostEqual(rmse, root_mean_squared_error(self.y_test, self.y_pred))

    def test_model_predict(self):
        # Test model prediction logic
        success, r2, rmse, predictions = model_predict(
            self.mock_model, 'mockModel', self.X_test, self.y_test
        )
        self.assertTrue(success)
        self.assertTrue(len(predictions), len(self.y_test))
        self.mock_model.predict.assert_called_once()

    def test_frequency_distribution(self):
        # Test frequency distribution
        freq_results = frequency_distribution(self.y_test, self.y_preds)

        # Check if all models have their counts
        self.assertIn('sgdLinearRegressorModel', freq_results)
        self.assertIn('randomForestRegressorModel', freq_results)
        self.assertIn('mlpRegressorModel', freq_results)
        self.assertIn('nfnnRegressorModel', freq_results)

        # Ensure counts add up to the length of the test data
        total_count = sum(freq_results.values())
        self.assertEqual(total_count, len(self.y_test))

if __name__ == '__main__':
    unittest.main()
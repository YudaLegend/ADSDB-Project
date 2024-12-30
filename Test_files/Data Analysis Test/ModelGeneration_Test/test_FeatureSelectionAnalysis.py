import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from Python_files.Data_Analysis.ModelGeneration_Zone.FeatureSelectionAnalysis import metric_results, predictor_target_split, create_cv_feature_selection_folder, nfnnRegressor_feature_predict, split_datasets, split_datasets_nfnn_results, cross_validation_nfnn_feature

class TestModelGeneration(unittest.TestCase):

    def setUp(self):
        # Mock sample data
        self.data = pd.DataFrame({
            'languages': [0.1, 0.2, 0.3],
            'genres': [0.1, 0.2, 0.3],
            'categories': [0.1, 0.2, 0.3],
            'game_popularity': [0, 0, 1],
            'game_active_players_2days': [1000, 2000, 1500],
            'recommendation_ratio': [1, 0, 0],
            'average_playtime': [10, 15, 20],
            'median_playtime': [5, 7, 10],
            'price_discount': [0.2, 0.1, 0.15],
            'game_satisfaction': [4.5, 4.7, 4.6],
        })
        self.mock_con = MagicMock()
        self.mock_con.execute.return_value.df.return_value = self.data

    def test_metric_results(self):
        y_test = np.array([4.5, 4.7, 4.6])
        y_pred = np.array([4.4, 4.8, 4.5])
        
        r2, mse, explained_variance = metric_results(y_test, y_pred)
        
        self.assertIsInstance(r2, float)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(explained_variance, float)

    def test_predictor_target_split(self):
        X, y = predictor_target_split(self.data, 1)
        self.assertEqual(X.shape[1], 9)  # 9 features in 'X'
        self.assertEqual(y.shape[0], len(self.data))  # Same number of rows as in 'data'
        
        X, y = predictor_target_split(self.data, 2)
        self.assertEqual(X.shape[1], 6)  # 6 features in 'X' when type is 2
        self.assertEqual(y.shape[0], len(self.data))

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_create_cv_feature_selection_folder(self, mock_makedirs, mock_exists):
        result = create_cv_feature_selection_folder()
        mock_exists.assert_called_once_with('Data Analysis/ModelGeneration Zone/Models/CrossValidation/NFNNRegressorModel_FeatureSelection')
        mock_makedirs.assert_called_once_with('Data Analysis/ModelGeneration Zone/Models/CrossValidation/NFNNRegressorModel_FeatureSelection/', exist_ok=True)
        self.assertTrue(result)

    @patch("torch.load")
    def test_nfnnRegressor_feature_predict(self, mock_torch_load):
        # Mocking the model loading and prediction
        mock_nfnn_regressor = MagicMock()
        mock_nfnn_regressor.predict.return_value = np.array([4.5, 4.6, 4.7])

        X_test = self.data[['languages', 'genres', 'categories', 'game_popularity', 'game_active_players_2days', 
        'recommendation_ratio', 'average_playtime', 'median_playtime', 'price_discount']]
        y_test = self.data['game_satisfaction']
        result, r2, mse, explained_variance, y_pred = nfnnRegressor_feature_predict(mock_nfnn_regressor, X_test, y_test)
     
        self.assertTrue(result)
        self.assertEqual(len(y_pred), 3)
        self.assertIsInstance(r2, float)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(explained_variance, float)


    
    def test_split_datasets(self):
        
        split_data = split_datasets(self.mock_con)
        
        # Ensure that there are 5 splits (as per your cross-validation logic)
        self.assertEqual(len(split_data), 5)
        
        # Ensure that each train and test dataset has the correct number of rows
        self.assertEqual(split_data[0][0].shape[0], len(self.data))  # Train set should have the same number of rows
        self.assertEqual(split_data[0][1].shape[0], len(self.data))  # Test set should have the same number of rows

    @patch("torch.load")
    def test_split_datasets_nfnn_results(self, mock_torch_load):
        split_datasets = [(self.data, self.data)]  # Mock data
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([4.5, 4.6, 4.7]))  # Mock predictions
        # When torch.load is called, return the mock model
        mock_torch_load.return_value = mock_model
        
        result, r2_scores, mse_scores, explained_variance_scores = split_datasets_nfnn_results(split_datasets)

        self.assertTrue(result)
        self.assertIn("estimator_0", r2_scores)
        self.assertIn("estimator_0", mse_scores)
        self.assertIn("estimator_0", explained_variance_scores)

    @patch("Python_files.DataAnalysis.ModelGeneration_Zone.FeatureSelectionAnalysis.NumericalFeatureNeuralNetwork")
    @patch("torch.save")
    def test_cross_validation_nfnn_feature(self, mock_torch_save, mock_nfnn):
        split_datasets = [(self.data, self.data)]  # Mock data
        
        # Mock the NFNN regressor
        mock_nfnn_regressor = MagicMock()
        mock_nfnn_regressor.fit.return_value = None  # Mock fit method to return nothing
        mock_nfnn_regressor.predict = MagicMock(return_value=np.array([4.5, 4.6, 4.7]))  # Mock predictions
        
        # Make sure that calling NumericalFeatureNeuralNetwork returns the mock regressor
        mock_nfnn.return_value = mock_nfnn_regressor


        result, r2_scores, mse_scores, explained_variance_scores = cross_validation_nfnn_feature(split_datasets)

        self.assertTrue(result)
        self.assertIn("estimator_0", r2_scores)
        self.assertIn("estimator_0", mse_scores)
        self.assertIn("estimator_0", explained_variance_scores)

if __name__ == '__main__':
    unittest.main()
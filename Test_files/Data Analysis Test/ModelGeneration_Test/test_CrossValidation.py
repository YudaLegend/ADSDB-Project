import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from Python_files.Data_Analysis.ModelGeneration_Zone.CrossValidation import cross_validation, save_split_sets, create_cross_validation_folder  # type: ignore

class TestModelGeneration(unittest.TestCase):

    def setUp(self):
        # Setup mock for duckdb connection
        self.mock_con = MagicMock()

        # Sample data for testing cross-validation
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.rand(100, 5), columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        self.y = pd.Series(np.random.rand(100))

    @patch('os.makedirs')
    def test_create_cross_validation_folder(self, mock_makedirs):
        # Test folder creation
        result = create_cross_validation_folder()
        self.assertTrue(result)

        # Check if os.makedirs was called with the expected arguments
        mock_makedirs.assert_any_call('Data Analysis/ModelGeneration Zone/Models/CrossValidation/', exist_ok=True)  # Check if base folder is created
        mock_makedirs.assert_any_call('Data Analysis/ModelGeneration Zone/Models/CrossValidation/SGDLinearRegressorModel/', exist_ok=True)
        mock_makedirs.assert_any_call('Data Analysis/ModelGeneration Zone/Models/CrossValidation/MLPRegressorModel/', exist_ok=True)
        mock_makedirs.assert_any_call('Data Analysis/ModelGeneration Zone/Models/CrossValidation/NFNNRegressorModel/', exist_ok=True)

    def test_cross_validation_success(self):
        # Mock the return of cross-validation
        mock_cv_results = {
            'sgdLinearRegressorModel': {
                'test_score': [0.5, 0.6, 0.7, 0.8, 0.9],
                'train_score': [0.8, 0.9, 0.85, 0.75, 0.9],
                'estimator': [MagicMock() for _ in range(5)],
                'indices': {
                    'train': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]],
                    'test': [[15, 16, 17], [18, 19, 20], [21, 22, 23], [24, 25, 26], [27, 28, 29]]
                }
            }
        }

        # Mocking the actual cross-validation results
        self.mock_con.execute.return_value = MagicMock()

        # Mock the cross-validation call
        with unittest.mock.patch('Python_files.DataAnalysis.ModelGeneration_Zone.CrossValidation.cross_validate', return_value=mock_cv_results):
            success, results = cross_validation(self.X, self.y)
        
        # Assert the function worked correctly
        self.assertTrue(success)
        self.assertIn('sgdLinearRegressorModel', results)

    def test_save_split_sets(self):
        # Mock the cv_results
        mock_cv_results = {
            'indices': {
                'train': [[0, 1, 2], [3, 4, 5]],
                'test': [[6, 7, 8], [9, 10, 11]]
            }
        }

        # Mock the database execution to verify table creation
        self.mock_con.execute.return_value = MagicMock()

        # Test the saving of split sets
        success = save_split_sets(mock_cv_results, self.X, self.y, self.mock_con)

        # Assert the result
        self.assertTrue(success)

        # Verify that the appropriate database commands were called
        self.mock_con.execute.assert_any_call("DROP TABLE IF EXISTS cross_validation_train_set_0")
        self.mock_con.execute.assert_any_call("CREATE TABLE IF NOT EXISTS cross_validation_train_set_0 AS SELECT * FROM train_set")
        self.mock_con.execute.assert_any_call("DROP TABLE IF EXISTS cross_validation_test_set_0")
        self.mock_con.execute.assert_any_call("CREATE TABLE IF NOT EXISTS cross_validation_test_set_0 AS SELECT * FROM test_set")

if __name__ == '__main__':
    unittest.main()
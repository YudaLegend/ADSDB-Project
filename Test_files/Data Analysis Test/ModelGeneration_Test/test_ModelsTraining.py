import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from Python_files.Data_Analysis.ModelGeneration_Zone.ModelsTraining import (create_model_folder, save_model, create_model)  # Replace `your_module` with the actual module name


class TestModelCreation(unittest.TestCase):

    def setUp(self):
        # Mock data for training
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.rand(100)

    def test_create_model_folder(self):
        # Test folder creation logic
        with patch("os.makedirs") as mock_makedirs:
            result = create_model_folder()
            self.assertTrue(result)
            mock_makedirs.assert_called_once()

    def test_create_model(self):
        result = create_model(self.X_train, self.y_train, 'NFNNRegressor')
        self.assertTrue(result[0])  # Success flag
        self.assertAlmostEqual(result[1], r2_score(self.y_train, result[3].predict(self.X_train)), places=4)
        self.assertAlmostEqual(result[2], mean_squared_error(self.y_train, result[3].predict(self.X_train)), places=4)

    @patch("joblib.dump")
    def test_save_model_joblib(self, mock_joblib_dump):
        mock_model = MagicMock()
        result = save_model(mock_model, "sgdLinearRegressorModel")
        self.assertTrue(result)
        mock_joblib_dump.assert_called_once()

    @patch("torch.save")
    def test_save_model_torch(self, mock_torch_save):
        mock_model = MagicMock()
        result = save_model(mock_model, "nfnnRegressorModel")
        self.assertTrue(result)
        mock_torch_save.assert_called_once()

if __name__ == "__main__":
    unittest.main()

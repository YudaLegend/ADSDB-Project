import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Python_files/Data_Management/Landing_Zone')))

import unittest
from unittest.mock import patch, MagicMock, call
import os
import shutil
from TemporalToPersistent import create_folders, move_files # type: ignore

class TestTemporalToPersistent(unittest.TestCase):

    @patch("os.makedirs")
    def test_create_folders(self, mock_makedirs):
        # Run create_folders function
        create_folders()

        # Check if os.makedirs was called with the expected folder paths
        mock_makedirs.assert_any_call('./Data Management/Formatted Zone', exist_ok=True)
        mock_makedirs.assert_any_call('./Data Management/Trusted Zone', exist_ok=True)
        mock_makedirs.assert_any_call('./Data Management/Exploitation Zone', exist_ok=True)
        mock_makedirs.assert_any_call('./Data Management/Landing Zone/Persistent', exist_ok=True)
        self.assertEqual(mock_makedirs.call_count, 4)


    @patch("os.listdir", return_value=["steam_data_detail_api_v1.json", "steam_spy_data_api_v1.json"])
    @patch("os.path.isfile", side_effect=lambda x: True)  # Mock to always return True for any path
    @patch("os.makedirs")
    @patch("shutil.copy")
    def test_move_files(self, mock_copy, mock_makedirs, mock_isfile, mock_listdir):
        # Run the move_files function
        result = move_files()

        # Assertions
        mock_listdir.assert_called_once_with('./Data Management/Landing Zone/Temporal')  # Check if source folder is listed

        # Verify folder creation calls (without file extensions)
        mock_makedirs.assert_has_calls([
            call('./Data Management/Landing Zone/Persistent\\steam_data_detail_api', exist_ok=True),
            call('./Data Management/Landing Zone/Persistent\\steam_spy_data_api', exist_ok=True)
        ], any_order=True)

        # Verify shutil.copy was called with the correct parameters
        mock_copy.assert_has_calls([
            call('./Data Management/Landing Zone/Temporal\\steam_data_detail_api_v1.json', './Data Management/Landing Zone/Persistent\\steam_data_detail_api\\steam_data_detail_api_v1.json'),
            call('./Data Management/Landing Zone/Temporal\\steam_spy_data_api_v1.json', './Data Management/Landing Zone/Persistent\\steam_spy_data_api\\steam_spy_data_api_v1.json')
        ], any_order=True)

        # Check that the function returns True if all operations succeed
        self.assertTrue(result)


    @patch("os.listdir", return_value=[])
    @patch("shutil.move")
    def test_move_files_no_files(self, mock_move, mock_listdir):
        # Run move_files function when there are no files
        move_files()

        # os.listdir should be called
        mock_listdir.assert_called_once_with('./Data Management/Landing Zone/Temporal')

        # shutil.move should not be called since no files are present
        mock_move.assert_not_called()

    @patch("os.listdir", side_effect=FileNotFoundError)
    def test_move_files_source_not_found(self, mock_listdir):
        # Run move_files function and check if it returns False on FileNotFoundError
        result = move_files()
        self.assertFalse(result, "Expected move_files to return False on FileNotFoundError")

        # os.listdir should be called once and raise the FileNotFoundError
        mock_listdir.assert_called_once_with('./Data Management/Landing Zone/Temporal')


if __name__ == "__main__":
    unittest.main()

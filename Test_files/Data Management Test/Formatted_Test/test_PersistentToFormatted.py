import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Python_files/Data_Management/Formatted_Zone')))

import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from PersistentToFormatted import ( # type: ignore
    read_json, saveIntoDB, createTable, formattedDatasets
)

class TestPersistentToFormatted(unittest.TestCase):

    @patch("builtins.open")
    @patch("json.load")
    def test_read_json_success(self, mock_json_load, mock_open):
        mock_json_load.return_value = {"key": "value"}
        result = read_json("valid_path.json")
        self.assertEqual(result, {"key": "value"})

    @patch("builtins.open")
    def test_read_json_failure(self, mock_open):
        mock_open.side_effect = IOError("File not found")
        result = read_json("invalid_path.json")
        self.assertFalse(result)

    @patch("pandas.DataFrame.to_sql")
    def test_save_into_db(self, mock_to_sql):
        # Mock DuckDB connection
        mock_con = MagicMock()
        mock_con.execute = MagicMock()
        # Sample data
        table_name = "test_table"
        version = 1
        formatted_data = [{"appid": 1, "publisher": "Pub1", "developer": "Dev1"}, {"appid": 2, "publisher": "Pub2", "developer": "Dev2"}]
        saveIntoDB(table_name, version, formatted_data, mock_con)
        mock_con.execute.assert_called_once()
        self.assertTrue(mock_con.execute.called)


    @patch("PersistentToFormatted.read_json")
    @patch("PersistentToFormatted.saveIntoDB")
    def test_create_table_success(self, mock_save, mock_read_json):
        mock_read_json.return_value = {
            "1": {"response": {"player_count": 100, "result": 1}}
        }
        mock_con = MagicMock()

        result = createTable("dummy_path.json", "steam_api_current_players_data", 1, mock_con)
        self.assertTrue(result)
        self.assertTrue(mock_save.called)

    @patch("PersistentToFormatted.read_json", return_value=False)
    def test_create_table_read_json_failure(self, mock_read_json):
        mock_con = MagicMock()
        result = createTable("invalid_path.json", "steam_api_current_players_data", 1, mock_con)
        self.assertFalse(result)



    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("PersistentToFormatted.createTable")
    def test_formatted_datasets_success(self, mock_create_table, mock_isdir, mock_listdir):
        # Mock the directory listing and structure
        mock_listdir.side_effect = [
            ["steam_api_app_details_data", "steam_api_steamspy_data"],  # Top-level directories in 'Persistent'
            ["steam_api_app_details_data_v1.json", "steam_api_app_details_data_v2.json"],  # Files in 'steam_api_app_details_data'
            ["steam_api_steamspy_data_v1.json"]  # Files in 'steam_api_steamspy_data'
        ]
        mock_isdir.side_effect = [True, True]  # Indicate both are directories
        mock_create_table.return_value = True  # Mock successful table creation

        # Mock connection object
        mock_con = MagicMock()

        # Call the function under test
        result = formattedDatasets(mock_con)

        # Assertions
        self.assertTrue(result)
        self.assertEqual(mock_create_table.call_count, 3)  # 2 files in 'steam_api_app_details_data' + 1 file in 'steam_api_steamspy_data'
        mock_create_table.assert_any_call('./Data Management/Landing Zone/Persistent/steam_api_app_details_data/steam_api_app_details_data_v1.json', 'steam_api_app_details_data', 1, mock_con)
        mock_create_table.assert_any_call('./Data Management/Landing Zone/Persistent/steam_api_app_details_data/steam_api_app_details_data_v2.json', 'steam_api_app_details_data', 2, mock_con)
        mock_create_table.assert_any_call('./Data Management/Landing Zone/Persistent/steam_api_steamspy_data/steam_api_steamspy_data_v1.json', 'steam_api_steamspy_data', 1, mock_con)

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("PersistentToFormatted.createTable")
    def test_formatted_datasets_failure(self, mock_create_table, mock_isdir, mock_listdir):
        # Mock the directory listing and structure
        mock_listdir.side_effect = [
            ["steam_api_app_details_data"],  # Top-level directory in 'Persistent'
            ["steam_api_app_details_data_v1.json", "steam_api_app_details_data_v2.json"]  # Files in 'steam_api_app_details_data'
        ]
        mock_isdir.side_effect = [True]  # 'steam_api_app_details_data' is a directory
        mock_create_table.side_effect = [True, False]  # First file succeeds, second fails

        # Mock connection object
        mock_con = MagicMock()

        # Call the function under test
        result = formattedDatasets(mock_con)

        # Assertions
        self.assertFalse(result)
        self.assertEqual(mock_create_table.call_count, 2)  # Attempted both files
        mock_create_table.assert_called_with('./Data Management/Landing Zone/Persistent/steam_api_app_details_data/steam_api_app_details_data_v2.json', 'steam_api_app_details_data', 2, mock_con)

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("PersistentToFormatted.createTable")
    def test_formatted_datasets_empty_directory(self, mock_create_table, mock_isdir, mock_listdir):
        # Mock the directory listing and structure
        mock_listdir.side_effect = [[]]  # No directories in 'Persistent'
        mock_isdir.return_value = False  # No directories found
        mock_create_table.return_value = False  # Default return for completeness

        # Mock connection object
        mock_con = MagicMock()

        # Call the function under test
        result = formattedDatasets(mock_con)

        # Assertions
        self.assertTrue(result)  # No errors should occur, function should return True
        mock_create_table.assert_not_called()  # No tables to create




















   
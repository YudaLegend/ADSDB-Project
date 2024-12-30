import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Python_files/Data_Management/Trusted_Zone')))
import ast
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from TrustedQualityExtractionTransformation import ( # type: ignore
    epic_games_extraction_transformation,
    steam_app_details_extraction_transformation,
    steam_spy_extraction_transformation,
    data_extraction_transformation,
    read_table,
    reload_table
)


class TestDataExtractionTransformation(unittest.TestCase):

    def test_read_table(self):
        # Mock connection and return value for execute
        mock_con = MagicMock()
        mock_df = pd.DataFrame({"appid": [1, 2], "publisher": ["a", "b"]})
        mock_con.execute.return_value.df.return_value = mock_df

        # Call read_table and verify results
        table_name = "test_table"
        result_df = read_table(mock_con, table_name)

        mock_con.execute.assert_called_once_with(f"SELECT * FROM {table_name};")
        self.assertTrue(result_df.equals(mock_df))

    def test_reload_table(self):
        # Mock connection
        mock_con = MagicMock()
        # Test DataFrame
        test_df = pd.DataFrame({"appid": [1, 2], "publisher": ["a", "c"]})

        # Call reload_table
        table_name = "test_table"
        reload_table(mock_con, table_name, test_df)

        # Verify the connection methods were called with correct SQL
        mock_con.execute.assert_any_call(f"DROP TABLE IF EXISTS {table_name};")
        mock_con.execute.assert_any_call(f"CREATE TABLE {table_name} AS SELECT * FROM df")


    @patch("TrustedQualityExtractionTransformation.read_table")
    @patch("TrustedQualityExtractionTransformation.reload_table")
    def test_epic_games_extraction_transformation(self, mock_reload_table, mock_read_table):
        # Mock DataFrame
        input_data = pd.DataFrame({
            "name": ["Game1", "Game2"],
            "seller": ['{"name": "Epic"}', '{"name": "Epic Games"}'],
            "customAttributes": [
                '[{"key": "publisherName", "value": "Epic Publisher"}, {"key": "developerName", "value": "Epic Dev"}]',
                '[{"key": "publisherName", "value": "Another Publisher"}]'
            ],
            "categories": ['[{"path": "games"}]', '[{"path": "apps"}]'],
            "effectiveDate": ['2022-01-01T12:00:00.000Z', ''],
        })
        
        expected_data = pd.DataFrame({
            "name": ["Game1", "Game2"],
            "seller": ["Epic", "Epic Games"],
            "categories": ["game", ""],
            "effectiveDate": ["2022-01-01", "UNKNOWN"],
            "publisherName": ["EPIC PUBLISHER", "ANOTHER PUBLISHER"],
            "developerName": ["EPIC DEV", "UNKNOWN"]
        })

        mock_read_table.return_value = input_data

        # Mock connection
        mock_con = MagicMock()

        # Call the function
        epic_games_extraction_transformation(mock_con)

        # Assertions
        mock_read_table.assert_called_once_with(mock_con, 'epic_games')
        mock_reload_table.assert_called_once()
        
        # Verify that reload_table was called with the extracted and transformated DataFrame
        extracted_transformated_df = mock_reload_table.call_args[0][2]
        pd.testing.assert_frame_equal(extracted_transformated_df, expected_data)

    @patch("TrustedQualityExtractionTransformation.read_table")
    @patch("TrustedQualityExtractionTransformation.reload_table")
    def test_steam_app_details_extraction_transformation(self, mock_reload_table, mock_read_table):
        # Mock DataFrame
        input_data = pd.DataFrame({
            "app_id": [1, 2],
            "developers": ['["Dev1", "Dev2"]', '["Dev3"]'],
            "publishers": ['["Pub1"]', '["Pub2", "Pub3"]'],
            "release_date": [{'date': '10 Jan, 2023'}, {'date': ''}],
            "recommendations": ['{"total": 1000}', '{"total": 500}'],
            "achievements": ['{"total": 50}', '{"total": 30}'],
            "metacritic": ['{"score": 80}', ''],
            "dlc": ['["DLC1", "DLC2"]', '["DLC3"]'],
            "platforms": [{"windows": True, "mac": True, "linux": True}, ''],
            "supported_languages": ['English, French, German', 'English-American, Spanish'],
            "categories": ['Category1, Category2', 'Category3'],
            "genres": ['Genre1, Genre2', '']
        })
        expected_data = pd.DataFrame({
            "app_id": [1, 2],
            "developers": ["DEV1, DEV2", "DEV3"],
            "publishers": ["PUB1", "PUB2, PUB3"],
            "release_date": ["2023-01-10", "UNKNOWN"],
            "recommendations": [1000, 500],
            "achievements": [50, 30],
            "dlc": [2, 1],
            "supported_languages": ["ENGLISH, FRENCH, GERMAN", "ENGLISH, SPANISH"],
            "categories": ['CATEGORY1, CATEGORY2', 'CATEGORY3'],
            "genres": ['GENRE1, GENRE2', 'UNKNOWN'],
            "metacritic_score": [80, 0],
            "windows_platform": [True, False],
            "linux_platform": [True, False],
            "mac_platform": [True, False]
        })

        mock_read_table.return_value = input_data

        # Mock connection
        mock_con = MagicMock()

        # Call the function
        steam_app_details_extraction_transformation(mock_con)

        # Assertions
        mock_read_table.assert_called_once_with(mock_con, 'steam_app_details')
        mock_reload_table.assert_called_once()
        
        # Verify that reload_table was called with the extracted and transformated DataFrame
        extracted_transformated_df = mock_reload_table.call_args[0][2]
        pd.testing.assert_frame_equal(extracted_transformated_df, expected_data)

    @patch("TrustedQualityExtractionTransformation.read_table")
    @patch("TrustedQualityExtractionTransformation.reload_table")
    def test_steam_spy_extraction_transformation(self, mock_reload_table, mock_read_table):
        # Mock DataFrame
        input_data = pd.DataFrame({
            "appid": [1, 2],
            "developer": ["Dev1", ""],
            "publisher": ["Pub1", ""],
            "languages": ["English, French", ""],
            "genre": ["Action", ""],
            "owners": ["1000..5000", "10000..50000"],
            "price": ["19.99", ""],
            "initialprice": ["39.99", ""]
        })
        expected_data = pd.DataFrame({
            "appid": [1, 2],
            "developer": ["DEV1", "UNKNOWN"],
            "publisher": ["PUB1", "UNKNOWN"],
            "languages": ["ENGLISH, FRENCH", "UNKNOWN"],
            "genre": ["ACTION", "UNKNOWN"],
            "owners": [3000.0, 30000.0],
            "price": ["19.99", 0],
            "initialprice": ["39.99", 0]
        })

        mock_read_table.return_value = input_data

        # Mock connection
        mock_con = MagicMock()

        # Call the function
        steam_spy_extraction_transformation(mock_con)

        # Assertions
        mock_read_table.assert_called_once_with(mock_con, 'steam_spy')
        mock_reload_table.assert_called_once()
        
        # Verify that reload_table was called with the extracted and transformated DataFrame
        extracted_transformated_df = mock_reload_table.call_args[0][2]
        pd.testing.assert_frame_equal(extracted_transformated_df, expected_data)


    @patch("TrustedQualityExtractionTransformation.epic_games_extraction_transformation", return_value = True)
    @patch("TrustedQualityExtractionTransformation.steam_app_details_extraction_transformation", return_value = True)
    @patch("TrustedQualityExtractionTransformation.steam_spy_extraction_transformation", return_value = True)
    def test_data_extraction_transformation_success(self, mock_epic_games, mock_steam_spy, mock_steam_details ):
        # Mock connection
        mock_con = MagicMock()

        # Call the function
        result = data_extraction_transformation(mock_con)

        # Assertions
        self.assertTrue(result)
        mock_epic_games.assert_called_once_with(mock_con)
        mock_steam_details.assert_called_once_with(mock_con)
        mock_steam_spy.assert_called_once_with(mock_con)

    
    @patch("TrustedQualityExtractionTransformation.steam_spy_extraction_transformation", side_effect=Exception("Steam Spy failed"))
    def test_data_extraction_transformation_failure(self, mock_steam_spy):
        # Mock connection
        mock_con = MagicMock()

        # Call the function
        result = data_extraction_transformation(mock_con)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

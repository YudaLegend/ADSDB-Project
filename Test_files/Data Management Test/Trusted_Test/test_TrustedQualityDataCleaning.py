import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Python_files/Data_Management/Trusted_Zone')))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import json
from TrustedQualityDataCleaning import ( # type: ignore
    read_table, reload_table, epic_games_data_cleaning, steam_current_players_cleaning,
    steam_app_details_cleaning, steam_spy_cleaning, data_cleaning
)


class TestDataCleaning(unittest.TestCase):
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

    @patch("TrustedQualityDataCleaning.read_table")
    @patch("TrustedQualityDataCleaning.reload_table")
    def test_epic_games_data_cleaning(self, mock_reload_table, mock_read_table):
        # Mock input DataFrame returned by read_table
        input_data = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Game1", "Game2", "Game3", "Game4"],
            "namespace": ["ns1", "ns2", "ns2", "ns3"],
            "url": ["url1", "url2", "url2", "url3"],
            "urlSlug": ["slug1", "slug2", "slug2", "slug3"],
            "keyImages": ["img1", "img2", "img2", "img3"],
            "items": ["item1", "item2", "item2", "item3"],
            "tags": ["tag1", "tag2", "tag2", "tag3"],
            "productSlug": ["pslug1", "pslug2", "pslug2", "pslug3"],
            "publisher": ["Pub1", "Pub2", "Pub2", "Pub3"]
        })
        mock_read_table.return_value = input_data

        # Expected output DataFrame after cleaning
        expected_data = pd.DataFrame({
            "name": ["Game1", "Game2", "Game3", "Game4"],
            "publisher": ["Pub1", "Pub2", "Pub2", "Pub3"]
        })

        # Mock connection object
        mock_con = MagicMock()

        # Call the function
        epic_games_data_cleaning(mock_con)

        # Assertions
        mock_read_table.assert_called_once_with(mock_con, 'epic_games')
        mock_reload_table.assert_called_once()

        # Verify that reload_table was called with the cleaned DataFrame
        cleaned_df = mock_reload_table.call_args[0][2]
        pd.testing.assert_frame_equal(cleaned_df, expected_data)



    @patch("TrustedQualityDataCleaning.read_table")
    @patch("TrustedQualityDataCleaning.reload_table")
    def test_steam_current_players_data_cleaning(self, mock_reload_table, mock_read_table):
        # Mock input DataFrame returned by read_table
        input_data = pd.DataFrame({
            "app_id": ["123", "456"],
            "player_count": [111, 222],
            "result": [1, 1]
        })
        mock_read_table.return_value = input_data

        # Expected output DataFrame after cleaning
        expected_data = pd.DataFrame({
            "app_id": [123, 456],
            "player_count": [111, 222],
        })

        # Mock connection object
        mock_con = MagicMock()

        # Call the function
        steam_current_players_cleaning(mock_con)

        # Assertions
        mock_read_table.assert_called_once_with(mock_con, 'steam_current_players')
        mock_reload_table.assert_called_once()

        # Verify that reload_table was called with the cleaned DataFrame
        cleaned_df = mock_reload_table.call_args[0][2]
        pd.testing.assert_frame_equal(cleaned_df, expected_data)


    @patch("TrustedQualityDataCleaning.read_table")
    @patch("TrustedQualityDataCleaning.reload_table")
    def test_steam_spy_data_cleaning(self, mock_reload_table, mock_read_table):
        # Mock input DataFrame returned by read_table
        input_data = pd.DataFrame({
            "appid": [123, 456, 123],
            "name": ["Game1", "Game2", "Game3"],
            "discount": [0, 0, 0],
            "developer": ["Dev1", "Dev2", "Dev1"],
            "publisher": ["Pub1", "Pub2", "Pub2"],
            "languages": ["English, French", "English, German", "English, Spanish"],
            "genre": ["Action, Adventures", "Action, RPG", "Action"],
            "owners": ["1..2", "2..3", "4..5"],
            "price": [1.99, 2.99, 3.99],
            "initialprice": [1.99, 2.99, 3.99],
            "score_rank": [1, 2, 3],
            "tags": ["", "", ""],
        })
        mock_read_table.return_value = input_data

        # Expected output DataFrame after cleaning
        expected_data = pd.DataFrame({
            "appid": [123, 456],
            "name": ["Game1", "Game2"],
            "discount": [0, 0],
            "developer": ["Dev1", "Dev2"],
            "publisher": ["Pub1", "Pub2"],
            "languages": ["English, French", "English, German"],
            "genre": ["Action, Adventures", "Action, RPG"],
            "owners": ["1..2", "2..3"],
            "price": [1.99, 2.99],
            "initialprice": [1.99, 2.99]
        })

        # Mock connection object
        mock_con = MagicMock()

        # Call the function
        steam_spy_cleaning(mock_con)

        # Assertions
        mock_read_table.assert_called_once_with(mock_con, 'steam_spy')
        mock_reload_table.assert_called_once()

        # Verify that reload_table was called with the cleaned DataFrame
        cleaned_df = mock_reload_table.call_args[0][2]
        pd.testing.assert_frame_equal(cleaned_df, expected_data)



    @patch("TrustedQualityDataCleaning.read_table")
    @patch("TrustedQualityDataCleaning.reload_table")
    def test_steam_app_details_data_cleaning(self, mock_reload_table, mock_read_table):
        # Mock input DataFrame returned by read_table
        input_data = pd.DataFrame({
            "steam_appid": [123, 456],
            "name": ["Game1", "Game2"],
            "detailed_description": ["testd1", "testd2"],
            "short_description": ["testsd1", "testsd2"],
            "about_the_game": ["testatg1", "testatg2"],
            "controller_support": ["full", ""],
            "header_image": ["testh1", "testh2"],
            "capsule_image": ["testc1", "testc2"],
            "capsule_imagev5": ["testc1v5", "testc2v5"],
            "website": ["testw1", "testw2"],
            "legal_notice": ["testl1", "testl2"],
            "pc_requirements": ["testpr1", "testpr2"],
            "mac_requirements": ["testmr1", "testmr2"],
            "linux_requirements": ["testlr1", "testlr2"],
            "screenshots": ["tests1", "tests2"],
            "movies": ["testm1", "testm2"],
            "ext_user_account_notice": ["teste1", "teste2"],
            "drm_notice": ["testd1", "testd2"],
            "support_info": ["testsi1", "testsi2"],
            "background": ["testsb1", "testsb2"],
            "background_raw": ["testsbr1", "testsbr2"],
            "reviews": ["testr1", "testr2"],
            "content_descriptors": ["testcd1", "testcd2"],
            "ratings": ["testra1", "testra2"],
            "demos": ["testde1", "testde2"],
            "packages": ["testpa1", "testpa2"],
            "package_groups": ["testpg1", "testpg2"],
            "fullgame": ["testfg1", "testfg2"],
            "developers": ["['Dev1']", "['Dev2']"],
            "publishers": ["[Pub1']", "['Pub2']"],
            "release_date": [{"date": "1 Jan, 2022"}, {"date": "2 Jan, 2022"}],
            "recommendations": ["{\"total\": 10}", "{\"total\": 20}"],
            "achievements": ["{\"total\": 5}", "{\"total\": 10}"],
            "metacritic": ["{\"score\": 80}", "{\"score\": 90}"],
            "platforms": ["", ""],
            "dlc": ["['DLC1', 'DLC2']", "['DLC3']"],
            "price_overview": ["{\"currency\": \"USD\", \"initial\": 1000, \"final\": 500}", "{\"currency\": \"USD\", \"initial\": 1000, \"final\": 500}"],
            "supported_languages": ["['English', 'French']", "['English', 'German']"],
            "categories": ["[{\"description\": \"Action\"}]", "[{\"description\": \"Adventure\"}]"],
            "genres": ["[{\"description\": \"Action\"}]", "[{\"description\": \"Adventure\"}]"]
        })
        mock_read_table.return_value = input_data

        # Expected output DataFrame after cleaning
        expected_data = pd.DataFrame({
            "steam_appid": [123, 456],
            "name": ["Game1", "Game2"],
            "detailed_description": ["testd1", "testd2"],
            "short_description": ["testsd1", "testsd2"],
            "about_the_game": ["testatg1", "testatg2"],
            "developers": ["['Dev1']", "['Dev2']"],
            "publishers": ["[Pub1']", "['Pub2']"],
            "release_date": [{"date": "1 Jan, 2022"}, {"date": "2 Jan, 2022"}],
            "recommendations": ["{\"total\": 10}", "{\"total\": 20}"],
            "achievements": ["{\"total\": 5}", "{\"total\": 10}"],
            "metacritic": ["{\"score\": 80}", "{\"score\": 90}"],
            "platforms": ["", ""],
            "dlc": ["['DLC1', 'DLC2']", "['DLC3']"],
            "price_overview": ["{\"currency\": \"USD\", \"initial\": 1000, \"final\": 500}", "{\"currency\": \"USD\", \"initial\": 1000, \"final\": 500}"],
            "supported_languages": ["['English', 'French']", "['English', 'German']"],
            "categories": ["[{\"description\": \"Action\"}]", "[{\"description\": \"Adventure\"}]"],
            "genres": ["[{\"description\": \"Action\"}]", "[{\"description\": \"Adventure\"}]"]
        })

        # Mock connection object
        mock_con = MagicMock()

        # Call the function
        steam_app_details_cleaning(mock_con)

        # Assertions
        mock_read_table.assert_called_once_with(mock_con, 'steam_app_details')
        mock_reload_table.assert_called_once()

        # Verify that reload_table was called with the cleaned DataFrame
        cleaned_df = mock_reload_table.call_args[0][2]
        pd.testing.assert_frame_equal(cleaned_df, expected_data)


    @patch("TrustedQualityDataCleaning.epic_games_data_cleaning", return_value=True)
    @patch("TrustedQualityDataCleaning.steam_current_players_cleaning", return_value=True)
    @patch("TrustedQualityDataCleaning.steam_app_details_cleaning", return_value=True)
    @patch("TrustedQualityDataCleaning.steam_spy_cleaning", return_value=True)
    def test_data_cleaning_success(self, mock_epic_games, mock_steam_current_players , mock_steam_app_details, mock_steam_spy):
        mock_con = MagicMock()
        result = data_cleaning(mock_con)
        self.assertTrue(result)
        mock_epic_games.assert_called_once()
        mock_steam_current_players.assert_called_once()
        mock_steam_app_details.assert_called_once()
        mock_steam_spy.assert_called_once()

    
    @patch("TrustedQualityDataCleaning.epic_games_data_cleaning", side_effect=Exception("Epic Games failed"))
    def test_data_cleaning_failure(self, mock_epic_games):
        mock_con = MagicMock()
        result = data_cleaning(mock_con)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
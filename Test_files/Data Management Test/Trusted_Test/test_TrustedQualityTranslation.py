import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Python_files/Data_Manament/Trusted_Zone')))

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from TrustedQualityTranslation import ( # type: ignore
    steam_app_details_translation,
    steam_spy_translation,
    data_translation,
    read_table,
    reload_table,
    translate_with_cache,
    translate_items,
    delete_languages_support
)


class TestDataTranslation(unittest.TestCase):
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

    def test_translate_with_cache(self):

        # Call translate_with_cache and assert results
        self.assertEqual(translate_with_cache("hola"), "hello")

    def test_translate_items(self):
        with patch("TrustedQualityTranslation.translate_with_cache", side_effect=lambda x: f"translated_{x}") as mock_translate:
            result = translate_items("item1, item2")
            self.assertEqual(result, "TRANSLATED_ITEM1, TRANSLATED_ITEM2")

    def test_delete_languages_support(self):
        result = delete_languages_support("English, LANGUAGES WITH FULL AUDIO SUPPORT")
        self.assertEqual(result, "English")

    @patch("TrustedQualityTranslation.read_table")
    @patch("TrustedQualityTranslation.reload_table")
    def test_steam_app_details_translation(self, mock_reload_table, mock_read_table):
        # Mock input DataFrame
        input_data = pd.DataFrame({
            "categories": ["[{'DESCRIPTION': 'Category 1'}, {'DESCRIPTION': 'Category 2'}]", ""],
            "genres": ["[{'DESCRIPTION': 'Genre 1'}, {'DESCRIPTION': 'Genre 2'}]", ""],
            "supported_languages": ["Inglés", "Espagnol"],
            "name": ["App1", "App2"],
            "detailed_description": ["Desc1", ""],
            "about_the_game": ["About1", ""],
            "short_description": ["Short1", ""]
        })

        expected_data = pd.DataFrame({
            "categories": ["CATEGORY 1, CATEGORY 2", ""],
            "genres": ["GENRE 1, GENRE 2", ""],
            "supported_languages": ["ENGLISH", "SPANISH"],
            "name": ["App1", "App2"],
            "detailed_description": ["Desc1", "UNKNOWN"],
            "about_the_game": ["About1", "UNKNOWN"],
            "short_description": ["Short1", "UNKNOWN"],
        })

        mock_read_table.return_value = input_data

        # Mock connection
        mock_con = MagicMock()

        # Call the function
        steam_app_details_translation(mock_con)

        # Assertions
        mock_read_table.assert_called_once_with(mock_con, 'steam_app_details')
        mock_reload_table.assert_called_once()

        # Verify that translated values were applied
        translated_df = mock_reload_table.call_args[0][2]
        pd.testing.assert_frame_equal(translated_df, expected_data)

    @patch("TrustedQualityTranslation.read_table")
    @patch("TrustedQualityTranslation.reload_table")
    def test_steam_spy_translation(self, mock_reload_table, mock_read_table):
        # Mock input DataFrame
        input_data = pd.DataFrame({
            "languages": ["English", "Espagnol"],
            "genre": ["Acción, RPG", "Adventure, Puzzle"],
            "name": ["Game1", ""],
            "discount": [20, ""]
        })

        expected_data = pd.DataFrame({
            "languages": ["ENGLISH", "SPANISH"],
            "genre": ["ACTION, RPG", "ADVENTURE, PUZZLE"],
            "name": ["Game1", "UNKNOWN"],
            "discount": [20, 0]
        })
        mock_read_table.return_value = input_data

        # Mock connection
        mock_con = MagicMock()

        # Call the function
        steam_spy_translation(mock_con)

        # Assertions
        mock_read_table.assert_called_once_with(mock_con, 'steam_spy')
        mock_reload_table.assert_called_once()

        # Verify that translated values were applied
        translated_df = mock_reload_table.call_args[0][2]
        pd.testing.assert_frame_equal(translated_df, expected_data)

    @patch("TrustedQualityTranslation.steam_app_details_translation", return_value=True)
    @patch("TrustedQualityTranslation.steam_spy_translation", return_value=True)
    def test_data_translation_success(self, mock_steam_spy, mock_steam_details):
        # Mock connection
        mock_con = MagicMock()

        # Call the function
        result = data_translation(mock_con)

        # Assertions
        self.assertTrue(result)
        mock_steam_details.assert_called_once_with(mock_con)
        mock_steam_spy.assert_called_once_with(mock_con)

    @patch("TrustedQualityTranslation.steam_spy_translation", side_effect=Exception("Steam Spy failed"))
    def test_data_translation_failure(self, mock_steam_spy):
        # Mock connection
        mock_con = MagicMock()

        # Call the function
        result = data_translation(mock_con)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

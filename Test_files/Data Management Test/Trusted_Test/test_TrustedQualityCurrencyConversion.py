import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Python_files/Data_Management/Trusted_Zone')))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import json
from TrustedQualityCurrencyConversion import ( # type: ignore
    read_table, reload_table, currency_conversion, convert_price,
    epic_games_currency_conversion, steam_app_details_currency_conversion,
    steam_spy_currency_conversion, data_currency_conversion
)


class TestDataCurrencyConversion(unittest.TestCase):


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


    @patch("TrustedQualityCurrencyConversion.CurrencyConverter")
    def test_currency_conversion(self, mock_currency_converter):
        result = currency_conversion(1000, 'USD')
        self.assertEqual(result, 9.21)


    def test_convert_price_valid_data(self):
        price_json = json.dumps({
            "totalPrice": {
                "originalPrice": "2000",
                "currencyCode": "USD"
            }
        })
        with patch("TrustedQualityCurrencyConversion.currency_conversion", return_value=20.0) as mock_currency_conversion:
            result = convert_price(price_json, "originalPrice")
            self.assertEqual(result, 20.0)
            mock_currency_conversion.assert_called_once_with(2000, "USD")

    def test_convert_price_invalid_data(self):
        price_json = "invalid json string"
        result = convert_price(price_json, "originalPrice")
        self.assertEqual(result, '')

    @patch("TrustedQualityCurrencyConversion.read_table")
    @patch("TrustedQualityCurrencyConversion.reload_table")
    @patch("TrustedQualityCurrencyConversion.convert_price")
    def test_epic_games_currency_conversion(self, mock_convert_price, mock_reload_table, mock_read_table):
        mock_df = pd.DataFrame({
            "price": ["{\"totalPrice\": {\"originalPrice\": 1000, \"discountPrice\": 1000, \"currencyCode\": \"USD\"}}",
                      "{\"totalPrice\": {\"originalPrice\": 2000, \"discountPrice\": 2000, \"currencyCode\": \"USD\"}}"
                    ]
        })
        mock_read_table.return_value = mock_df
        mock_convert_price.side_effect = [10.0, 10.0, 20.0, 20.0]

        mock_con = MagicMock()
        epic_games_currency_conversion(mock_con)

        self.assertEqual(mock_df["originalPrice"].iloc[0], 10.0)
        self.assertEqual(mock_df["discountPrice"].iloc[0], 20.0)
        self.assertEqual(mock_df["originalPrice"].iloc[1], 10.0)
        self.assertEqual(mock_df["discountPrice"].iloc[1], 20.0)
        mock_reload_table.assert_called_once()

    @patch("TrustedQualityCurrencyConversion.read_table")
    @patch("TrustedQualityCurrencyConversion.reload_table")
    def test_steam_app_details_currency_conversion(self, mock_reload_table, mock_read_table):
        mock_df = pd.DataFrame({
            "price_overview": ['{"currency": "USD", "initial": "2000", "final": "1000"}', 
                               '{"currency": "EUR", "initial": "3000", "final": "2500"}']
        })
        mock_read_table.return_value = mock_df

        with patch("TrustedQualityCurrencyConversion.currency_conversion", side_effect=[20.0, 10.0, 30.0, 25.0]):
            mock_con = MagicMock()
            steam_app_details_currency_conversion(mock_con)

            self.assertEqual(mock_df["initial_price"].iloc[0], 20.0)
            self.assertEqual(mock_df["final_price"].iloc[0], 10.0)
            self.assertEqual(mock_df["initial_price"].iloc[1], 30.0)
            self.assertEqual(mock_df["final_price"].iloc[1], 25.0)
            mock_reload_table.assert_called_once()

    @patch("TrustedQualityCurrencyConversion.read_table")
    @patch("TrustedQualityCurrencyConversion.reload_table")
    @patch("TrustedQualityCurrencyConversion.currency_conversion")
    def test_steam_spy_currency_conversion(self, mock_currency_conversion, mock_reload_table, mock_read_table):
        mock_df = pd.DataFrame({
            "price": [19.99, 29.99],
            "initialprice": [39.99, 49.99]
        })
        mock_read_table.return_value = mock_df
        mock_currency_conversion.side_effect = [19.99, 29.99, 39.99, 49.99]

        mock_con = MagicMock()
        steam_spy_currency_conversion(mock_con)

        self.assertEqual(mock_df["price"].iloc[0], 19.99)
        self.assertEqual(mock_df["initialprice"].iloc[0], 39.99)
        mock_reload_table.assert_called_once()

    @patch("TrustedQualityCurrencyConversion.epic_games_currency_conversion", return_value=True)
    @patch("TrustedQualityCurrencyConversion.steam_app_details_currency_conversion", return_value=True)
    @patch("TrustedQualityCurrencyConversion.steam_spy_currency_conversion", return_value=True)
    def test_data_currency_conversion_success(self, mock_steam_spy, mock_steam_details, mock_epic_games):
        mock_con = MagicMock()
        result = data_currency_conversion(mock_con)
        self.assertTrue(result)
        mock_epic_games.assert_called_once()
        mock_steam_details.assert_called_once()
        mock_steam_spy.assert_called_once()

    @patch("TrustedQualityCurrencyConversion.epic_games_currency_conversion", side_effect=Exception("Epic Games failed"))
    def test_data_currency_conversion_failure(self, mock_epic_games):
        mock_con = MagicMock()
        result = data_currency_conversion(mock_con)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
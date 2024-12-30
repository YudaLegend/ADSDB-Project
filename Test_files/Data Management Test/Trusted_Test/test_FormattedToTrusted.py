import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Python_files/Data_Management/Trusted_Zone')))

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from FormattedToTrusted import UnifyTables # type: ignore

class TestFormattedToTrusted(unittest.TestCase):

    @patch("pandas.DataFrame")
    def test_UnifyTables_success(self, mock_dataframe):
        mock_con = MagicMock()
        mock_trusted_conn = MagicMock()

        # Mock DataFrames for consistency
        mock_con.execute.return_value.df.side_effect = [
            pd.DataFrame(columns=["id", "name", "release_year", "type", "genre", "price"]),  
            pd.DataFrame({"id": [1, 2], "name": ["game1", "game2"], "release_year": [2021, 2022], "type": ["game", "dlc"], "genre": ["indie, action", "adventure"], "price": [10.99, 20.99]})
        ]
        
        # Mock register on trusted_conn
        mock_trusted_conn.register = MagicMock()

        # Run UnifyTables
        result = UnifyTables(
            tables_to_combine=["table1", "table2"],
            table_name="combined_table",
            con=mock_con,
            trusted_conn=mock_trusted_conn
        )

        # Assert expectations
        mock_con.execute.assert_any_call("SELECT * FROM table1 LIMIT 0")
        self.assertTrue(mock_trusted_conn.register.called)
        mock_trusted_conn.register.assert_called_with('combined_data_df', mock_dataframe.return_value)
        self.assertTrue(result)

    @patch("pandas.DataFrame")
    def test_UnifyTables_empty_tables_list(self, mock_dataframe):
        mock_con = MagicMock()
        mock_trusted_conn = MagicMock()

        # Run UnifyTables with an empty list and check the return value
        result = UnifyTables(
            tables_to_combine=[],  
            table_name="combined_table",
            con=mock_con,
            trusted_conn=mock_trusted_conn
        )
        self.assertFalse(result, "Expected UnifyTables to return False on empty tables list")

    @patch("pandas.DataFrame")
    def test_UnifyTables_register_failure(self, mock_dataframe):
        mock_con = MagicMock()
        mock_trusted_conn = MagicMock()

        # Mock DataFrame and simulate register failure
        mock_con.execute.return_value.df.side_effect = [
            pd.DataFrame(columns=["id", "name", "release_year", "type", "genre", "price"]),  
            pd.DataFrame({"id": [1, 2], "name": ["game1", "game2"], "release_year": [2021, 2022], "type": ["game", "dlc"], "genre": ["indie, action", "adventure"], "price": [10.99, 20.99]})
        ]
        mock_trusted_conn.register.side_effect = Exception("Registration Failed")

        # Run UnifyTables and expect False due to register failure
        result = UnifyTables(
            tables_to_combine=["table1", "table2"],
            table_name="combined_table",
            con=mock_con,
            trusted_conn=mock_trusted_conn
        )
        self.assertFalse(result, "Expected UnifyTables to return False on register failure")

if __name__ == "__main__":
    unittest.main()

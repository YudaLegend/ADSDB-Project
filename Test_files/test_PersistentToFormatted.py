import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Python_files')))

import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from PersistentToFormatted import (
    open_file, saveIntoDBCurrentPlayer, steamCurrentPlayerDataset,
    saveIntoDBSteamGameInfo, steamGameDetailsDataset,
    saveIntoDBSteamSpy, steamSpyDataset,
    saveIntoDBEpicGames, epicGamesDataset
)

class TestPersistentToFormatted(unittest.TestCase):

    @patch("builtins.open")
    @patch("json.load")
    def test_open_file_success(self, mock_json_load, mock_open):
        mock_json_load.return_value = {"key": "value"}
        result = open_file("valid_path.json")
        self.assertEqual(result, {"key": "value"})

    @patch("builtins.open")
    def test_open_file_failure(self, mock_open):
        mock_open.side_effect = IOError("File not found")
        result = open_file("invalid_path.json")
        self.assertFalse(result)

    @patch("PersistentToFormatted.open_file")
    @patch("pandas.DataFrame")
    def test_saveIntoDBCurrentPlayer(self, mock_dataframe, mock_open_file):
        mock_open_file.return_value = {"1": {"response": {"player_count": 100, 'result': 1}}}
        mock_con = MagicMock()
        mock_dataframe.return_value = pd.DataFrame([{"app_id": "1", "player_count": 100, "result": 1}])
        result = saveIntoDBCurrentPlayer(mock_con, "path.json", "1.0")
        self.assertTrue(result)

    @patch("PersistentToFormatted.open_file")
    @patch("pandas.DataFrame")
    def test_saveIntoDBSteamGameInfo(self, mock_dataframe, mock_open_file):
        mock_open_file.return_value = {"1": {"success": True, "data": {"type": 'game', "name": "game1", "steam_appid": 1, "is_free": True}}}
        mock_con = MagicMock()
        mock_dataframe.return_value = pd.DataFrame([{"type": "game", "name": "game1", "steam_appid": 1, "is_free": True}])
        result = saveIntoDBSteamGameInfo(mock_con, "path.json", "1.0")
        self.assertTrue(result)

    @patch("PersistentToFormatted.open_file")
    @patch("pandas.DataFrame")
    def test_saveIntoDBSteamSpy(self, mock_dataframe, mock_open_file):
        mock_open_file.return_value = {"1": {"appid": 1, "name": "game1", "developer": "Dev1", "publisher": "Pub1"}}
        mock_con = MagicMock()
        mock_dataframe.return_value = pd.DataFrame([{"app_id": 1, "game1": "value", "developer": "Dev1", "publisher": "Pub1"}])
        result = saveIntoDBSteamSpy(mock_con, "path.json", "1.0")
        self.assertTrue(result)

    @patch("PersistentToFormatted.open_file")
    @patch("pandas.DataFrame")
    def test_saveIntoDBEpicGames(self, mock_dataframe, mock_open_file):
        mock_open_file.return_value = {"1": {"data": {"Catalog": {"searchStore": {"elements": [{"title": "game1", "id": "abc123", "namespace": "cb23c857ec0d42d89b4be34d11302959"}]}}}}}
        mock_con = MagicMock()
        mock_dataframe.return_value = pd.DataFrame([{"key": "value"}])
        result = saveIntoDBEpicGames(mock_con, "dummy_path.json", "1.0")
        self.assertTrue(result)

    @patch("PersistentToFormatted.saveIntoDBCurrentPlayer", return_value=True)
    @patch("os.listdir", return_value=["steam_current_player_api_v1.json", "steam_current_player_api_v2.json"])
    def test_steamCurrentPlayerDataset(self, mock_listdir, mock_save):
        mock_con = MagicMock()
        result = steamCurrentPlayerDataset(mock_con)
        self.assertTrue(result)

    @patch("PersistentToFormatted.saveIntoDBSteamGameInfo", return_value=True)
    @patch("os.listdir", return_value=["steam_game_info_api_v1.json", "steam_game_info_api_v2.json"])
    def test_steamGameDetailsDataset(self, mock_listdir, mock_save):
        mock_con = MagicMock()
        result = steamGameDetailsDataset(mock_con)
        self.assertTrue(result)

    @patch("PersistentToFormatted.saveIntoDBSteamSpy", return_value=True)
    @patch("os.listdir", return_value=["steam_spy_api_v1.json", "steam_spy_api_v2.json"])
    def test_steamSpyDataset(self, mock_listdir, mock_save):
        mock_con = MagicMock()
        result = steamSpyDataset(mock_con)
        self.assertTrue(result)

    @patch("PersistentToFormatted.saveIntoDBEpicGames", return_value=True)
    @patch("os.listdir", return_value=["epic_games_api_v1.json", "epic_games_api_v2.json"])
    def test_epicGamesDataset(self, mock_listdir, mock_save):
        mock_con = MagicMock()
        result = epicGamesDataset(mock_con)
        self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()

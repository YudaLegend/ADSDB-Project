import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Python_files')))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import json
from TrustedQuality import (
    extract_name, extract_customAttributes_info, extract_categories, convert_price,
    data_formatted, epicGamesQuality, steamCurrentPlayersQuality, unify_languages_format,
    translate_languages, translate_genres_spy, steamSpyQuality, convert_to_string,
    data_formatted_detail, currencyConvert, translate_with_cache, translate_text, getTranslated,
    unify_languages_format_detail, steamGameDetailsQuality
)

class TestTrustedQuality(unittest.TestCase):

    # Tests for Epic Games Table Functions
    def test_extract_name_success(self):
        json_value = json.dumps({"name": "Seller1"})
        result = extract_name(json_value)
        self.assertEqual(result, "Seller1")

    def test_extract_name_invalid_json(self):
        result = extract_name("invalid json")
        self.assertEqual(result, "")

    def test_extract_customAttributes_info_success(self):
        json_value = json.dumps([{"key": "publisherName", "value": "Pub1"}])
        result = extract_customAttributes_info(json_value, "publisherName")
        self.assertEqual(result, "Pub1")

    def test_extract_customAttributes_info_invalid_json(self):
        result = extract_customAttributes_info("invalid json", "publisherName")
        self.assertEqual(result, "")

    def test_extract_categories_success(self):
        json_value = json.dumps([{"path": "games"}, {"path": "music"}])
        result = extract_categories(json_value)
        self.assertEqual(result, "game")

    def test_extract_categories_invalid_json(self):
        result = extract_categories("invalid json")
        self.assertEqual(result, "")

    @patch("TrustedQuality.CurrencyConverter.convert", return_value=10.0)
    def test_convert_price_success(self, mock_convert):
        json_value = json.dumps({"totalPrice": {"originalPrice": 1000, "currencyCode": "USD"}})
        result = convert_price(json_value, "originalPrice")
        self.assertEqual(result, 0.1)
    
    @patch("TrustedQuality.CurrencyConverter.convert", return_value=10.0)
    def test_convert_price_invalid_json(self, mock_convert):
        json_value = json.dumps('invalid json')
        result = convert_price(json_value, "originalPrice")
        self.assertEqual(result, '')

    def test_data_formatted_success(self):
        date_str = "2022-01-01T00:00:00.000Z"
        result = data_formatted(date_str)
        self.assertEqual(result, "2022-01-01")

    def test_data_formatted_invalid_date(self):
        date_str = "invalid date"
        result = data_formatted(date_str)
        self.assertEqual(result, "")

    # Tests for Epic Games Quality Function
    
    def test_epicGamesQuality(self):
        mock_con = MagicMock()
        mock_con.execute.return_value.df.return_value = pd.DataFrame({
            "id": [1],    
            "namespace": ["test_namespace"],
            "url": ["test_url"],
            "urlSlug": ["test_url_slug"],
            "keyImages": ["test_key_images"],
            "items": ["test_items"],
            "tags": ["test_tags"],
            "productSlug": ["test_product_slug"],
            "seller": ["{\"name\": \"Test Seller\"}"],
            "customAttributes": ["[{\"key\": \"publisherName\", \"value\": \"Test Publisher\"}]"],
            "categories": ["[{\"path\": \"games\"}]"],
            "price": ["{\"totalPrice\": {\"originalPrice\": 1000, \"currencyCode\": \"USD\"}}"],
            "effectiveDate": ["2022-01-01T00:00:00.000Z"]
        })
        result = epicGamesQuality(mock_con)
        self.assertTrue(result)

    # Tests for Steam Current Players Quality
    def test_steamCurrentPlayersQuality(self):
        mock_con = MagicMock()
        mock_con.execute.return_value.df.return_value = pd.DataFrame({
            "app_id": ["123", "456"],
            "player_count": [111, 222],
            "result": [1, 1]
        })
        result = steamCurrentPlayersQuality(mock_con)
        self.assertTrue(result)

    # Tests for Steam Spy Functions
    def test_unify_languages_format(self):
        text = "English, French, German"
        result = unify_languages_format(text)
        self.assertEqual(result, "English, French, German")

    def test_translate_languages(self):
        result = translate_languages("Anglais, Français")
        self.assertEqual(result, "English, French")

    def test_translate_genres_spy(self):
        result = translate_genres_spy("Acción, Aventuras")
        self.assertEqual(result, "Action, Adventures")


    # Tests for Steam Spy Quality
    def test_steamSpyQuality(self):
        mock_con = MagicMock()
        mock_con.execute.return_value.df.return_value = pd.DataFrame({
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
        result = steamSpyQuality(mock_con)
        self.assertTrue(result)

    # Tests for Steam Games Details Functions
    def test_convert_to_string_success(self):
        list_str = "['Action', 'Adventure']"
        result = convert_to_string(list_str)
        self.assertEqual(result, "Action, Adventure")
    
    def test_convert_to_string_failure(self):
        empty_str = ""
        result = convert_to_string(empty_str)
        self.assertEqual(result, "UNKNOWN")

    def test_data_formatted_detail_success(self):
        date_json = {"date": "1 Jan, 2022"}
        result = data_formatted_detail(date_json)
        self.assertEqual(result, "2022-01-01")
    
    def test_data_formatted_detail_failure(self):
        date_json = {"not_date": "invalid date"}
        result = data_formatted_detail(date_json)
        self.assertEqual(result, "UNKNOWN")

    @patch("TrustedQuality.CurrencyConverter.convert", return_value=1.0)
    def test_currencyConvert(self, mock_convert):
        df = pd.DataFrame({"price_overview": ["{\"currency\": \"USD\", \"initial\": 1000, \"final\": 500}"]})
        initial, final = currencyConvert(df)
        self.assertEqual(initial, [0.01])
        self.assertEqual(final, [0.01])

    @patch('TrustedQuality.translation_cache', new_callable=dict)
    def test_translate_with_cache(self, mock_cache):
        
        translate_with_cache.__globals__["translation_cache"] = {}
        result = translate_with_cache("Hola")
        self.assertEqual(result, "Hello")

    def test_translate_text(self):
        text = [{'description': 'Hola'}]
        result = translate_text(text)
        self.assertEqual(result, "Hello")

    def test_getTranslated(self):
        genre = "[{\"description\": \"Action\"}]"
        category = "[{\"description\": \"Adventure\"}]"
        result = getTranslated(genre, category)
        self.assertEqual(result, ("Action", "Adventure"))

    def test_unify_languages_format_detail(self):
        text = "English, French, German, Spanish, languages with full audio support"
        result = unify_languages_format_detail(text)
        self.assertEqual(result, "English, French, German, Spanish")

    # Tests for Steam Game Details Table Quality
    @patch("TrustedQuality.getTranslated", return_value=("Action", "Adventure"))
    def test_steamGameDetailsQuality(self, mock_translate):
        mock_con = MagicMock()
        mock_con.execute.return_value.df.return_value = pd.DataFrame({
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
        result = steamGameDetailsQuality(mock_con)
        self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()

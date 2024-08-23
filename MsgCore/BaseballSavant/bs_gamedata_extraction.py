import json
import pandas as pd


class BaseballGameProcessor:
    def __init__(self, json_file):
        self.json_file = json_file

    @staticmethod
    def extract_game_data(game_json):
        game_id = game_json['scoreboard']['gamePk']
        game_date = game_json['gameDate']
        game_data = []

        base_record = {
            'game_id': game_id,
            'game_date': game_date,
            'game_status_code': game_json['game_status_code'],
            'game_status': game_json['game_status'],
            'gameday_type': game_json['gamedayType'],
        }

        # Extract scoreboard details
        scoreboard = game_json.get('scoreboard', {})
        if scoreboard:
            for key, value in scoreboard.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        base_record[f"scoreboard_{key}_{sub_key}"] = sub_value
                else:
                    base_record[f"scoreboard_{key}"] = value

        # Extract WPA details
        stats = scoreboard.get('stats', {})
        if stats:
            for stat_type, stat_values in stats.items():
                if isinstance(stat_values, list):
                    for i, item in enumerate(stat_values):
                        for item_key, item_value in item.items():
                            base_record[f"stats_{stat_type}_{i}_{item_key}"] = item_value
                elif isinstance(stat_values, dict):
                    for sub_key, sub_value in stat_values.items():
                        base_record[f"stats_{stat_type}_{sub_key}"] = sub_value
                else:
                    base_record[f"stats_{stat_type}"] = stat_values

        game_data.append(base_record)
        return game_data

    def load_json(self):
        with open(self.json_file) as f:
            return json.load(f)

    def process_games(self):
        games_json = self.load_json()
        all_game_data = []
        for game_json in games_json:
            game_data = self.extract_game_data(game_json)
            all_game_data.extend(game_data)

        df = pd.DataFrame(all_game_data)
        return df
import json
import pandas as pd


class bs_playerdata_extraction:
    def __init__(self, json_file):
        self.json_file = json_file

    @staticmethod
    def extract_player_data(game_json):
        game_id = game_json['scoreboard']['gamePk']
        game_date = game_json['gameDate']
        player_data = []

        for player_id, player_info in game_json['players'].items():
            base_record = {
                'game_id': game_id,
                'date': game_date,
                'player_id': player_info['person']['id'],
                'player_name': player_info['person']['fullName'],
                'jersey_number': player_info.get('jerseyNumber'),
                'position_code': player_info['position']['code'],
                'position_name': player_info['position']['name'],
                'batting_order': player_info.get('battingOrder'),
                'status_code': player_info['status']['code'],
                'status_description': player_info['status']['description'],
                'team_id': player_info['parentTeamId'],
            }

            # Flatten stats, season_stats, game_status, all_positions
            if 'stats' in player_info:
                for stat_type, stats in player_info['stats'].items():
                    for stat_key, stat_value in stats.items():
                        base_record[f"{stat_type}_{stat_key}"] = stat_value

            if 'seasonStats' in player_info:
                for stat_type, stats in player_info['seasonStats'].items():
                    for stat_key, stat_value in stats.items():
                        base_record[f"season_{stat_type}_{stat_key}"] = stat_value

            if 'gameStatus' in player_info:
                for status_key, status_value in player_info['gameStatus'].items():
                    base_record[f"game_status_{status_key}"] = status_value

            if 'allPositions' in player_info:
                for i, position in enumerate(player_info['allPositions']):
                    for pos_key, pos_value in position.items():
                        base_record[f"position_{i}_{pos_key}"] = pos_value

            player_data.append(base_record)

        return player_data

    def load_json(self):
        with open(self.json_file) as f:
            return json.load(f)

    def process_games(self):
        games_json = self.load_json()
        all_player_data = []
        for game_json in games_json:
            player_data = self.extract_player_data(game_json)
            all_player_data.extend(player_data)

        df = pd.DataFrame(all_player_data)
        return df

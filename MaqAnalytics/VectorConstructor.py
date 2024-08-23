import pandas as pd


class VectorConstructor:
    def __init__(self, player_df):
        self.player_df = player_df

    @staticmethod
    def extract_team_players(game_json):
        home_batters = []
        home_pitchers = []
        away_batters = []
        away_pitchers = []

        players = game_json['players']

        for player_id, player_info in players.items():
            team_id = player_info['parentTeamId']
            position = player_info['position']['code']

            if team_id == game_json['teams']['home']['id']:
                if position == 'P':
                    home_pitchers.append(player_id)
                else:
                    home_batters.append(player_id)
            elif team_id == game_json['teams']['away']['id']:
                if position == 'P':
                    away_pitchers.append(player_id)
                else:
                    away_batters.append(player_id)

        return {
            "home_batters": home_batters,
            "home_pitchers": home_pitchers,
            "away_batters": away_batters,
            "away_pitchers": away_pitchers
        }

    def fetch_player_stats(self, start, end, player_ids):
        # Convert date columns to datetime
        self.player_df['date'] = pd.to_datetime(self.player_df['date'])
        # Filter by date range
        player_df = self.player_df[(self.player_df['date'] >= start) & (self.player_df['date'] <= end)]
        # Get stats from relevant players
        return player_df[player_df['player_id'].isin(player_ids)]

    # TODO: Finish normalization logic (particularly for EWMA)
    def normalize_player_stats(self, start_date, end_date, player_ids, use_exponential_weight=True, alpha=0.5):
        # Fetch the filtered stats for the given players within the date range
        player_stats = self.fetch_player_stats(start_date, end_date, player_ids)

        # Define columns that should not be averaged (non-stat columns)
        non_stat_prefixes = ["season_", "game_status_", "position_"]
        base_columns = ['game_id', 'date', 'player_id', 'player_name', 'jersey_number',
                        'position_code', 'position_name', 'batting_order',
                        'status_code', 'status_description', 'team_id']

        # Identify stat columns by excluding non-stat columns
        stat_columns = [
            col for col in player_stats.columns
            if not any(col.startswith(prefix) for prefix in non_stat_prefixes) and col not in base_columns
        ]

        # Group by player_id and calculate the average or exponentially weighted average for these columns only
        if use_exponential_weight:
            # Exponentially weighted mean
            normalized_stats = player_stats.groupby('player_id')[stat_columns].apply(
                lambda x: x.sort_values('date').ewm(alpha=alpha).mean()
            )
        else:
            # Regular mean
            normalized_stats = player_stats.groupby('player_id')[stat_columns].mean()

        # Combine the normalized stats with non-stat columns like player_id, player_name, etc.
        non_stat_data = player_stats.groupby('player_id')[base_columns].first().reset_index()

        # Merge the non-stat columns back with the normalized stats
        final_normalized_stats = pd.merge(non_stat_data, normalized_stats, on='player_id')

        return final_normalized_stats

    # TODO: Include moneyline data in feature vector
    # TODO: Tack on target variable (win-loss binary / total runs / run differential / etc.)
    @staticmethod
    def construct_game_vector(home_batters_df, home_pitcher_df, away_batters_df, away_pitcher_df):
        # Filter columns for batters (batting and fielding stats only)
        batting_stat_columns = [col for col in home_batters_df.columns if col.startswith('batting') or col.startswith('fielding')]
        home_batters_df = home_batters_df[batting_stat_columns + ['at_bats', 'player_id']]
        away_batters_df = away_batters_df[batting_stat_columns + ['at_bats', 'player_id']]

        # Filter columns for pitchers (pitching stats only)
        pitching_stat_columns = [col for col in home_pitcher_df.columns if col.startswith('pitching')]
        home_pitcher_df = home_pitcher_df[pitching_stat_columns]
        away_pitcher_df = away_pitcher_df[pitching_stat_columns]

        # Ensure that only numeric columns are processed
        home_batters_df = home_batters_df.select_dtypes(include='number')
        away_batters_df = away_batters_df.select_dtypes(include='number')
        home_pitcher_df = home_pitcher_df.select_dtypes(include='number')
        away_pitcher_df = away_pitcher_df.select_dtypes(include='number')

        # Calculate the weighted average of the batting stats for the home team
        home_weighted_avg = (home_batters_df.drop(columns=['at_bats', 'player_id'])  # Drop non-stat columns
                             .multiply(home_batters_df['at_bats'], axis=0)  # Multiply by at_bats
                             .sum() / home_batters_df['at_bats'].sum())  # Weighted average

        # Calculate the weighted average of the batting stats for the away team
        away_weighted_avg = (away_batters_df.drop(columns=['at_bats', 'player_id'])  # Drop non-stat columns
                             .multiply(away_batters_df['at_bats'], axis=0)  # Multiply by at_bats
                             .sum() / away_batters_df['at_bats'].sum())  # Weighted average

        # Average the pitching stats for the home team
        home_pitching_avg = home_pitcher_df.mean()

        # Average the pitching stats for the away team
        away_pitching_avg = away_pitcher_df.mean()

        # Concatenate the weighted averages and the pitching stats into a single vector for each team
        home_vector = pd.concat([home_weighted_avg, home_pitching_avg], axis=0)
        away_vector = pd.concat([away_weighted_avg, away_pitching_avg], axis=0)

        # Concatenate the home and away vectors into a single vector
        game_vector = pd.concat([home_vector, away_vector], axis=0)

        return game_vector

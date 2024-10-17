import asyncio
from datetime import datetime, timedelta
import pandas as pd
from OddsBlaze import OddsBlazeAPI
from BaseballSavant.bs_hist_gamelogs import bs_hist_gamelogs
from BaseballSavant.bs_range_gamelogs import bs_range_gamelogs


class VectorConstructor:
    def __init__(self, player_df, odds_api=OddsBlazeAPI(), moneylines_df=None):
        self.player_df = player_df
        self.odds_api = odds_api
        self.moneylines_df = moneylines_df
        self.abb_dict = {
            "ARI": "ARI",  # Arizona Diamondbacks
            "ATL": "ATL",  # Atlanta Braves
            "BAL": "BAL",  # Baltimore Orioles
            "BOS": "BOS",  # Boston Red Sox
            "CHC": "CHN",  # Chicago Cubs
            "CWS": "CHA",  # Chicago White Sox
            "CIN": "CIN",  # Cincinnati Reds
            "CLE": "CLE",  # Cleveland Indians
            "COL": "COL",  # Colorado Rockies
            "DET": "DET",  # Detroit Tigers
            "HOU": "HOU",  # Houston Astros
            "KC": "KCA",  # Kansas City Royals
            "LAA": "ANA",  # Los Angeles Angels
            "LAD": "LAN",  # Los Angeles Dodgers
            "MIA": "MIA",  # Miami Marlins
            "MIL": "MIL",  # Milwaukee Brewers
            "MIN": "MIN",  # Minnesota Twins
            "NYM": "NYN",  # New York Mets
            "NYY": "NYA",  # New York Yankees
            "OAK": "OAK",  # Oakland Athletics
            "PHI": "PHI",  # Philadelphia Phillies
            "PIT": "PIT",  # Pittsburgh Pirates
            "SD": "SDN",  # San Diego Padres
            "SF": "SFN",  # San Francisco Giants
            "SEA": "SEA",  # Seattle Mariners
            "STL": "SLN",  # St. Louis Cardinals
            "TB": "TBA",  # Tampa Bay Rays
            "TEX": "TEX",  # Texas Rangers
            "TOR": "TOR",  # Toronto Blue Jays
            "WSH": "WAS"  # Washington Nationals
        }
        self.retrosheet_field_names = [
            "game_id", "date", "game_number", "appearance_date", "team_id", "player_id",
            "batting_order", "batting_order_sequence", "home_flag", "opponent_id",
            "park_id",
            # Batting stats
            "B_G", "B_PA", "B_AB", "B_R", "B_H", "B_TB", "B_2B", "B_3B",
            "B_HR", "B_HR4", "B_RBI", "B_GW", "B_BB", "B_IBB", "B_SO",
            "B_GDP", "B_HP", "B_SH", "B_SF", "B_SB", "B_CS", "B_XI",
            "B_G_DH", "B_G_PH", "B_G_PR",
            # Pitching stats
            "P_G", "P_GS", "P_CG", "P_SHO", "P_GF", "P_W", "P_L",
            "P_SV", "P_OUT", "P_TBF", "P_AB", "P_R", "P_ER", "P_H",
            "P_TB", "P_2B", "P_3B", "P_HR", "P_HR4", "P_BB",
            "P_IBB", "P_SO", "P_GDP", "P_HP", "P_SH", "P_SF",
            "P_XI", "P_WP", "P_BK", "P_IR", "P_IRS", "P_GO",
            "P_AO", "P_PITCH", "P_STRIKE",
            # Fielding stats for P, C, 1B, 2B, 3B, SS, LF, CF, RF
            # Pitcher
            "F_P_G", "F_P_GS", "F_P_OUT", "F_P_TC", "F_P_PO",
            "F_P_A", "F_P_E", "F_P_DP", "F_P_TP",
            # Catcher
            "F_C_G", "F_C_GS", "F_C_OUT", "F_C_TC", "F_C_PO",
            "F_C_A", "F_C_E", "F_C_DP", "F_C_TP", "F_C_PB", "F_C_IX",
            # First Baseman
            "F_1B_G", "F_1B_GS", "F_1B_OUT", "F_1B_TC", "F_1B_PO",
            "F_1B_A", "F_1B_E", "F_1B_DP", "F_1B_TP",
            # Second Baseman
            "F_2B_G", "F_2B_GS", "F_2B_OUT", "F_2B_TC", "F_2B_PO",
            "F_2B_A", "F_2B_E", "F_2B_DP", "F_2B_TP",
            # Third Baseman
            "F_3B_G", "F_3B_GS", "F_3B_OUT", "F_3B_TC", "F_3B_PO",
            "F_3B_A", "F_3B_E", "F_3B_DP", "F_3B_TP",
            # Shortstop
            "F_SS_G", "F_SS_GS", "F_SS_OUT", "F_SS_TC", "F_SS_PO",
            "F_SS_A", "F_SS_E", "F_SS_DP", "F_SS_TP",
            # Left Fielder
            "F_LF_G", "F_LF_GS", "F_LF_OUT", "F_LF_TC", "F_LF_PO",
            "F_LF_A", "F_LF_E", "F_LF_DP", "F_LF_TP",
            # Center Fielder
            "F_CF_G", "F_CF_GS", "F_CF_OUT", "F_CF_TC", "F_CF_PO",
            "F_CF_A", "F_CF_E", "F_CF_DP", "F_CF_TP",
            # Right Fielder
            "F_RF_G", "F_RF_GS", "F_RF_OUT", "F_RF_TC", "F_RF_PO",
            "F_RF_A", "F_RF_E", "F_RF_DP", "F_RF_TP"
        ]

    def parse_game_details(self, game_json):
        # Extract relevant details from the game JSON
        game_date = game_json.get('gameDate')
        home_team_info = game_json['scoreboard']['defense']['team']
        away_team_info = game_json['scoreboard']['offense']['team']

        # Convert date to a datetime object
        game_date = datetime.strptime(game_date, "%m/%d/%Y").strftime('%Y-%m-%d')

        # Extract team abbreviations
        home_team_abbr = home_team_info['name']
        away_team_abbr = away_team_info['name']

        return game_date, home_team_abbr, away_team_abbr

    async def fetch_game_odds(self, game_json):
        # Parse game details
        game_date, home_team, away_team = self.parse_game_details(game_json)

        # Fetch moneyline data for the relevant league and date
        moneyline_data = await self.odds_api.get_game_moneyline_data('mlb', region='us', price_format='american')

        # Filter the results for the specific game
        game_odds = moneyline_data[
            (moneyline_data['home_team'] == self.abb_dict.get(home_team)) &
            (moneyline_data['away_team'] == self.abb_dict.get(away_team)) &
            (moneyline_data['commence_time'].str.contains(game_date))
            ]

        return game_odds

    def calculate_average_moneyline(self, moneylines_df):
        # Convert price column to numeric, ignoring errors for non-numeric values
        moneylines_df['price'] = pd.to_numeric(moneylines_df['price'], errors='coerce')

        # Group by game_id and team, then calculate the mean price for both home and away teams
        average_prices = moneylines_df.groupby(['game_id', 'home_team', 'away_team'])['price'].mean().reset_index()

        # Merge the average prices back into the original dataframe
        moneylines_df = pd.merge(moneylines_df, average_prices, on=['game_id', 'home_team', 'away_team'],
                                 suffixes=('', '_avg'))

        return moneylines_df

    @staticmethod
    def calculate_implied_odds(moneyline):
        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return -moneyline / (-moneyline + 100)

    def integrate_moneylines_and_odds(self, game_json):
        # Fetch game odds for the specific game
        game_odds = asyncio.run(self.fetch_game_odds(game_json))

        # Calculate average moneylines
        moneylines_df = game_odds.copy()  # Work with the fetched data
        moneylines_df = self.calculate_average_moneyline(moneylines_df)

        # Calculate implied odds and add them to the dataframe
        moneylines_df['home_team_implied_odds'] = moneylines_df['price_avg'].apply(self.calculate_implied_odds)
        moneylines_df['away_team_implied_odds'] = moneylines_df['price_avg'].apply(self.calculate_implied_odds)

        return moneylines_df

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

    # TODO: Change moneyline data source from OddsBlaze (live) to SportsBookReview (Historical)
    def construct_game_vector(self, game_json, home_batters_df, home_pitcher_df, away_batters_df, away_pitcher_df, moneylines_df):
        # Filter columns for batters (batting and fielding stats only)
        batting_stat_columns = [col for col in home_batters_df.columns if
                                col.startswith('batting') or col.startswith('fielding')]
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

        game_vector['home_team_implied_odds'] = moneylines_df['home_team_implied_odds'].iloc[0]
        game_vector['away_team_implied_odds'] = moneylines_df['away_team_implied_odds'].iloc[0]

        home_team_score = game_json['scoreboard']['defense']['score']
        away_team_score = game_json['scoreboard']['offense']['score']
        game_vector['home_team_won'] = 1 if home_team_score > away_team_score else 0
        return game_vector

    async def construct_all_game_vectors(self, date_ranges=None):
        # Initialize the game logs fetching class
        game_logs_fetcher = bs_range_gamelogs()

        # Fetch all game logs for the given date ranges
        game_jsons = game_logs_fetcher.get_gamelogs_for_date_ranges(date_ranges=date_ranges)

        all_game_vectors = []

        # Iterate over each game JSON
        for game_json in game_jsons:
            # Parse the game details
            game_date, home_team_abbr, away_team_abbr = self.parse_game_details(game_json)

            # Fetch and process moneyline data for the game
            moneylines_df = await self.fetch_game_odds(game_json)
            moneylines_df = self.calculate_average_moneyline(moneylines_df)
            moneylines_df['home_team_implied_odds'] = moneylines_df['price_avg'].apply(self.calculate_implied_odds)
            moneylines_df['away_team_implied_odds'] = moneylines_df['price_avg'].apply(self.calculate_implied_odds)

            # Extract player data up until this point in time
            player_data_timeframe = datetime.strptime(game_date, '%Y-%m-%d') - timedelta(days=365)  # Example: 1 year of data before the game
            player_ids = self.extract_team_players(game_json)
            home_batters_df = self.fetch_player_stats(player_data_timeframe, game_date, player_ids['home_batters'])
            home_pitcher_df = self.fetch_player_stats(player_data_timeframe, game_date, player_ids['home_pitchers'])
            away_batters_df = self.fetch_player_stats(player_data_timeframe, game_date, player_ids['away_batters'])
            away_pitcher_df = self.fetch_player_stats(player_data_timeframe, game_date, player_ids['away_pitchers'])

            # Construct the game vector for this game
            game_vector = self.construct_game_vector(home_batters_df, home_pitcher_df, away_batters_df, away_pitcher_df, moneylines_df, game_json)
            all_game_vectors.append(game_vector)

        # Combine all vectors into a single DataFrame
        all_game_vectors_df = pd.DataFrame(all_game_vectors)

        return all_game_vectors_df

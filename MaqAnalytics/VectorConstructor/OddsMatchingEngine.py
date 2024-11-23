import pandas as pd
import asyncio
from datetime import datetime, timedelta


# from SportsBookReview import SportsbookReviewScraper
# from bs_retrosheet_converter import SavantRetrosheetConverter

class OddsMatchingEngine:
    def __init__(self, moneylines_df, stat_df=None, player_df=None):
        """
        Initializes the VectorConstructor with necessary components.

        Parameters:
            moneylines_df (pd.DataFrame): DataFrame containing historical moneyline data. If None, it will be fetched using the scraper.
            stat_df (pd.DataFrame): DataFrame containing feature vectors based on player statistics (processed / pipelined)
            player_df (pd.DataFrame): DataFrame containing player statistics (unprocessed / not pipelined)
        """
        self.moneylines_df = moneylines_df  # or self.scrape_moneylines()
        self.stat_df = stat_df
        self.player_df = player_df
        # self.sportsbook_scraper = sportsbook_scraper
        # self.savant_converter = SavantRetrosheetConverter("01/01/2024")

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

    @staticmethod
    def parse_game_details(game_json):
        """
        Extracts relevant details from the game JSON.

        Parameters:
            game_json (dict): JSON data for the game.

        Returns:
            tuple: (game_date, home_team_abbr, away_team_abbr)
        """
        # Extract game date
        game_date_str = game_json.get('gameDate')
        game_date = datetime.strptime(game_date_str, "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d')

        # Extract team abbreviations
        home_team_abbr = game_json['scoreboard']['teams.json']['home']['team']['abbreviation']
        away_team_abbr = game_json['scoreboard']['teams.json']['away']['team']['abbreviation']

        return game_date, home_team_abbr, away_team_abbr

    @staticmethod
    def standardize_team_abbr(abbr):
        abbr_mapping = {
            'ARI': 'ARI', 'ARZ': 'ARI', 'AZ': 'ARI',
            'ATL': 'ATL',
            'BAL': 'BAL',
            'BOS': 'BOS',
            'CHC': 'CHC', 'CHN': 'CHC',
            'CWS': 'CWS', 'CHW': 'CWS', 'CHA': 'CWS',
            'CIN': 'CIN',
            'CLE': 'CLE',
            'COL': 'COL',
            'DET': 'DET',
            'HOU': 'HOU',
            'KC': 'KC', 'KCA': 'KC', 'KAN': 'KC',
            'LAA': 'LAA', 'ANA': 'LAA',
            'LAD': 'LAD', 'LAN': 'LAD',
            'MIA': 'MIA',
            'MIL': 'MIL',
            'MIN': 'MIN',
            'NYM': 'NYM', 'NYN': 'NYM',
            'NYY': 'NYY', 'NYA': 'NYY',
            'OAK': 'OAK',
            'PHI': 'PHI',
            'PIT': 'PIT',
            'SD': 'SD', 'SDP': 'SD',
            'SF': 'SF', 'SFN': 'SF',
            'SEA': 'SEA',
            'STL': 'STL', 'SLN': 'STL',
            'TB': 'TB', 'TBR': 'TB', 'TBA': 'TB',
            'TEX': 'TEX',
            'TOR': 'TOR',
            'WAS': 'WSH', 'WSH': 'WSH'
        }
        return abbr_mapping.get(abbr, abbr)

    @staticmethod
    def convert_odds_to_numeric(odds):
        odds = str(odds)
        if odds.startswith('+'):
            return int(odds[1:])
        elif odds.startswith('-'):
            return -int(odds[1:])
        else:
            return int(odds)

    @staticmethod
    def calculate_implied_odds(odds):
        """
        Calculates the implied probability from moneyline odds.

        Parameters:
            odds (float): Moneyline odds.

        Returns:
            float: Implied probability.
        """
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)

    def prepare_and_calculate_odds(self):
        """
        Processes the moneylines dataframe to pair the odds for each game into a single row.
        """
        # Standardize team abbreviations
        self.moneylines_df['team'] = self.moneylines_df['team'].apply(self.standardize_team_abbr)
        self.moneylines_df['opponent'] = self.moneylines_df['opponent'].apply(self.standardize_team_abbr)

        # Convert date to datetime
        self.moneylines_df['date'] = pd.to_datetime(self.moneylines_df['date'])

        # Remove percentage sign and convert wager_percentage to numeric
        self.moneylines_df['wager_percentage'] = self.moneylines_df['wager_percentage'].str.replace('%', '').astype(
            float)

        # Convert odds to numeric
        self.moneylines_df['numeric_odds'] = self.moneylines_df['odds'].apply(self.convert_odds_to_numeric)

        # Calculate implied odds
        self.moneylines_df['implied_odds'] = self.moneylines_df['numeric_odds'].apply(self.calculate_implied_odds)

        # Create a unique game identifier
        self.moneylines_df.dropna(subset=['team', 'opponent'], inplace=True)
        self.moneylines_df['game_id'] = self.moneylines_df.apply(
            lambda
                row: f"{row['date'].strftime('%Y-%m-%d')}_{sorted([row['team'], row['opponent']])[0]}_{sorted([row['team'], row['opponent']])[1]}",
            axis=1
        )

        # Group by game_id
        grouped = self.moneylines_df.groupby('game_id')

        # Initialize list to collect game odds
        game_odds_list = []

        # For each game, create a single row with home and away odds
        for game_id, group in grouped:
            date = group['date'].iloc[0]
            teams = group['team'].unique()
            if len(teams) != 2:
                # Skip games that don't have two teams.json
                continue
            team1, team2 = teams

            # Get data for both teams.json
            team1_rows = group[group['team'] == team1]
            team2_rows = group[group['team'] == team2]

            # Calculate averages across sportsbooks for both teams.json
            team1_avg_odds = team1_rows['numeric_odds'].mean()
            team1_avg_implied_odds = team1_rows['implied_odds'].mean()
            team1_avg_wager_percentage = team1_rows['wager_percentage'].mean()

            team2_avg_odds = team2_rows['numeric_odds'].mean()
            team2_avg_implied_odds = team2_rows['implied_odds'].mean()
            team2_avg_wager_percentage = team2_rows['wager_percentage'].mean()

            # **Calculate the bookmaker's vig**
            sum_implied_odds = team1_avg_implied_odds + team2_avg_implied_odds
            vig = sum_implied_odds - 1

            # **Adjust implied probabilities to sum to 1**
            team1_adjusted_implied_odds = team1_avg_implied_odds / sum_implied_odds
            team2_adjusted_implied_odds = team2_avg_implied_odds / sum_implied_odds

            # Create a dictionary for this game
            game_odds = {
                'game_id': game_id,
                'date': date,
                'team1': team1,
                'team2': team2,
                'team1_avg_odds': team1_avg_odds,
                'team1_avg_implied_odds': team1_adjusted_implied_odds,  # Use adjusted implied odds
                'team1_avg_wager_percentage': team1_avg_wager_percentage,
                'team2_avg_odds': team2_avg_odds,
                'team2_avg_implied_odds': team2_adjusted_implied_odds,  # Use adjusted implied odds
                'team2_avg_wager_percentage': team2_avg_wager_percentage,
                'vig': vig  # Add the vig to the game dictionary
            }

            game_odds_list.append(game_odds)

        # Convert list to DataFrame
        game_odds_df = pd.DataFrame(game_odds_list)

        return game_odds_df

    def match_game_moneylines_pipelined(self):
        """
        Matches and appends moneyline odds, their implied probabilities, and the bookmaker's vig to the stat_df.
        """
        # Prepare odds and calculate implied probabilities
        game_odds_df = self.prepare_and_calculate_odds()

        # Convert dates to datetime for accurate merging
        self.stat_df['Game_Date'] = pd.to_datetime(self.stat_df['Game_Date'])

        # Standardize team abbreviations in stat_df
        self.stat_df['Home_Team_Abbr'] = self.stat_df['Home_Team_Abbr'].apply(self.standardize_team_abbr)
        self.stat_df['Away_Team_Abbr'] = self.stat_df['Away_Team_Abbr'].apply(self.standardize_team_abbr)

        # Prepare for merging
        merged_df = self.stat_df.copy()

        # Function to find matching odds for each game
        def find_matching_odds(row):
            date = row['Game_Date']
            home_team = row['Home_Team_Abbr']
            away_team = row['Away_Team_Abbr']

            # Create game_id
            game_id = f"{date.strftime('%Y-%m-%d')}_{sorted([home_team, away_team])[0]}_{sorted([home_team, away_team])[1]}"

            # Try to find matching game in game_odds_df
            odds_row = game_odds_df[game_odds_df['game_id'] == game_id]

            if odds_row.empty:
                return pd.Series({
                    'home_odds': None,
                    'away_odds': None,
                    'home_implied_odds': None,
                    'away_implied_odds': None,
                    'home_wager_percentage': None,
                    'away_wager_percentage': None,
                    'vig': None
                })

            odds_row = odds_row.iloc[0]

            # Determine which team is home and assign odds accordingly
            if odds_row['team1'] == home_team:
                home_odds = odds_row['team1_avg_odds']
                home_implied_odds = odds_row['team1_avg_implied_odds']
                home_wager_percentage = odds_row['team1_avg_wager_percentage']
                away_odds = odds_row['team2_avg_odds']
                away_implied_odds = odds_row['team2_avg_implied_odds']
                away_wager_percentage = odds_row['team2_avg_wager_percentage']
            elif odds_row['team2'] == home_team:
                home_odds = odds_row['team2_avg_odds']
                home_implied_odds = odds_row['team2_avg_implied_odds']
                home_wager_percentage = odds_row['team2_avg_wager_percentage']
                away_odds = odds_row['team1_avg_odds']
                away_implied_odds = odds_row['team1_avg_implied_odds']
                away_wager_percentage = odds_row['team1_avg_wager_percentage']
            else:
                # Teams do not match, cannot assign odds
                return pd.Series({
                    'home_odds': None,
                    'away_odds': None,
                    'home_implied_odds': None,
                    'away_implied_odds': None,
                    'home_wager_percentage': None,
                    'away_wager_percentage': None,
                    'vig': None
                })

            # Include the vig
            vig = odds_row['vig']

            return pd.Series({
                'home_odds': home_odds,
                'away_odds': away_odds,
                'home_implied_odds': home_implied_odds,
                'away_implied_odds': away_implied_odds,
                'home_wager_percentage': home_wager_percentage,
                'away_wager_percentage': away_wager_percentage,
                'vig': vig
            })

        # Apply the function to each row
        odds_data = merged_df.apply(find_matching_odds, axis=1)

        # Concatenate the odds data with the merged_df
        merged_df = pd.concat([merged_df, odds_data], axis=1)

        return merged_df

    def calculate_average_moneyline(self, moneylines_df):
        """
        Calculates the average moneyline odds for home and away teams.json.

        Parameters:
            moneylines_df (pd.DataFrame): DataFrame containing moneyline data for a specific game.

        Returns:
            pd.DataFrame: DataFrame with average moneyline odds and implied probabilities.
        """
        # Convert price column to numeric, ignoring errors for non-numeric values
        moneylines_df['odds'] = pd.to_numeric(moneylines_df['odds'], errors='coerce')

        # Drop rows with NaN odds
        moneylines_df = moneylines_df.dropna(subset=['odds'])

        # Calculate average odds for home and away teams.json
        average_odds = moneylines_df.groupby(['team', 'opponent'])['odds'].mean().reset_index()

        # Calculate implied probabilities
        average_odds['home_team_implied_odds'] = average_odds['odds'].apply(self.calculate_implied_odds)
        average_odds['away_team_implied_odds'] = average_odds['odds'].apply(self.calculate_implied_odds)

        return average_odds

    @staticmethod
    def extract_team_players(game_json):
        """
        Extracts player IDs for home and away teams.json, segregating batters and pitchers.

        Parameters:
            game_json (dict): JSON data for the game.

        Returns:
            dict: Dictionary containing lists of home batters, home pitchers, away batters, away pitchers.
        """
        home_batters = []
        home_pitchers = []
        away_batters = []
        away_pitchers = []

        players = game_json['boxscore']['teams.json']['home'].get('players', {})
        for player_key, player_info in players.items():
            position = player_info.get('position', {}).get('code', '')
            player_id = player_info.get('person', {}).get('id')
            if position == 'P':
                home_pitchers.append(player_id)
            else:
                home_batters.append(player_id)

        players = game_json['boxscore']['teams.json']['away'].get('players', {})
        for player_key, player_info in players.items():
            position = player_info.get('position', {}).get('code', '')
            player_id = player_info.get('person', {}).get('id')
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
        """
        Fetches player statistics within a date range.

        Parameters:
            start (datetime): Start date.
            end (datetime): End date.
            player_ids (list): List of player IDs.

        Returns:
            pd.DataFrame: DataFrame containing statistics for the specified players.
        """
        # Ensure 'date' column is datetime
        self.player_df['date'] = pd.to_datetime(self.player_df['date'])

        # Filter by date range and player IDs
        player_stats = self.player_df[
            (self.player_df['date'] >= start) &
            (self.player_df['date'] <= end) &
            (self.player_df['player_id'].isin(player_ids))
            ]

        return player_stats

    def construct_game_vector(self, game_json, home_batters_df, home_pitcher_df, away_batters_df, away_pitcher_df,
                              moneylines_df):
        """
        Constructs a feature vector for a single game.

        Parameters:
            game_json (dict): JSON data for the game.
            home_batters_df (pd.DataFrame): DataFrame containing home batters' stats.
            home_pitcher_df (pd.DataFrame): DataFrame containing home pitchers' stats.
            away_batters_df (pd.DataFrame): DataFrame containing away batters' stats.
            away_pitcher_df (pd.DataFrame): DataFrame containing away pitchers' stats.
            moneylines_df (pd.DataFrame): DataFrame containing moneyline odds for the game.

        Returns:
            dict: Dictionary representing the feature vector for the game.
        """
        # Handle cases where no moneyline data is found
        if moneylines_df.empty:
            print(f"No moneyline data found for game_id {game_json['scoreboard']['gamePk']}. Skipping this game.")
            return None

        # Calculate weighted average of batting stats for home team
        home_weighted_avg = self.calculate_weighted_average(home_batters_df, 'at_bats')

        # Calculate weighted average of batting stats for away team
        away_weighted_avg = self.calculate_weighted_average(away_batters_df, 'at_bats')

        # Calculate average pitching stats for home team
        home_pitching_avg = home_pitcher_df.mean(numeric_only=True)

        # Calculate average pitching stats for away team
        away_pitching_avg = away_pitcher_df.mean(numeric_only=True)

        # Combine all features into a single dictionary
        feature_vector = {}

        # Home team features
        for stat, value in home_weighted_avg.items():
            feature_vector[f'home_{stat}'] = value
        for stat, value in home_pitching_avg.items():
            feature_vector[f'home_{stat}'] = value

        # Away team features
        for stat, value in away_weighted_avg.items():
            feature_vector[f'away_{stat}'] = value
        for stat, value in away_pitching_avg.items():
            feature_vector[f'away_{stat}'] = value

        # Moneyline features
        feature_vector['home_team_implied_odds'] = moneylines_df['home_team_implied_odds'].iloc[0]
        feature_vector['away_team_implied_odds'] = moneylines_df['away_team_implied_odds'].iloc[0]

        # Target variable
        home_team_score = game_json['scoreboard']['teams.json']['home']['score']
        away_team_score = game_json['scoreboard']['teams.json']['away']['score']
        feature_vector['home_team_won'] = 1 if home_team_score > away_team_score else 0

        return feature_vector

    @staticmethod
    def calculate_weighted_average(df: pd.DataFrame, weight_column: str) -> dict:
        """
        Calculates the weighted average of statistics based on a specified weight column.

        Parameters:
            df (pd.DataFrame): DataFrame containing player statistics.
            weight_column (str): Column name to be used as weights.

        Returns:
            dict: Dictionary of weighted average statistics.
        """
        if df.empty or weight_column not in df.columns:
            return {}

        # Select numeric columns for calculation
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        numeric_cols.remove(weight_column) if weight_column in numeric_cols else None

        # Calculate weighted average
        weighted_avg = {}
        total_weight = df[weight_column].sum()
        if total_weight == 0:
            # Avoid division by zero
            for col in numeric_cols:
                weighted_avg[col] = 0
        else:
            for col in numeric_cols:
                weighted_avg[col] = (df[col] * df[weight_column]).sum() / total_weight

        return weighted_avg

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime


class OddsMatchingEngine:
    def __init__(self, moneylines_df, stat_df=None, player_df=None):
        self.moneylines_df = moneylines_df
        self.stat_df = stat_df
        self.player_df = player_df

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
        odds = str(odds).strip()
        negative = False
        if odds.startswith('+'):
            odds_value = odds[1:]
        elif odds.startswith('-'):
            odds_value = odds[1:]
            negative = True
        else:
            odds_value = odds

        try:
            odds_int = int(odds_value)
            if negative:
                return -odds_int
            else:
                return odds_int
        except ValueError:
            # Handle invalid odds strings
            return np.nan

    @staticmethod
    def calculate_implied_odds(odds):
        """
        Calculates the implied probability from moneyline odds.

        Parameters:
            odds (float): Moneyline odds.

        Returns:
            float: Implied probability.
        """
        if pd.isna(odds):
            return np.nan
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
        self.moneylines_df['wager_percentage'] = self.moneylines_df['wager_percentage'].str.replace('%', '')
        self.moneylines_df['wager_percentage'] = self.moneylines_df['wager_percentage'].replace('-', np.nan)
        self.moneylines_df['wager_percentage'] = pd.to_numeric(
            self.moneylines_df['wager_percentage'],
            errors='coerce'
        )

        # Handle NaN values
        self.moneylines_df['wager_percentage'].fillna(0.5, inplace=True)

        # Convert odds to numeric
        self.moneylines_df['numeric_odds'] = self.moneylines_df['odds'].apply(self.convert_odds_to_numeric)

        # Drop rows with NaN 'numeric_odds'
        self.moneylines_df.dropna(subset=['numeric_odds'], inplace=True)

        # Calculate implied odds
        self.moneylines_df['implied_odds'] = self.moneylines_df['numeric_odds'].apply(self.calculate_implied_odds)

        # Create a unique game identifier
        self.moneylines_df.dropna(subset=['team', 'opponent'], inplace=True)
        self.moneylines_df['game_id'] = self.moneylines_df.apply(
            lambda row: f"{row['date'].strftime('%Y-%m-%d')}_{sorted([row['team'], row['opponent']])[0]}_{sorted([row['team'], row['opponent']])[1]}",
            axis=1
        )

        # Group by game_id
        grouped = self.moneylines_df.groupby('game_id')

        # Initialize list to collect game odds
        game_odds_list = []

        # Get total number of games for progress bar
        total_games = self.moneylines_df['game_id'].nunique()

        # For each game, create a single row with home and away odds
        for game_id, group in tqdm(grouped, total=total_games, desc='Processing games'):
            date = group['date'].iloc[0]
            teams = group['team'].unique()
            if len(teams) != 2:
                # Skip games that don't have two teams
                continue
            team1, team2 = teams

            # Get data for both teams
            team1_rows = group[group['team'] == team1]
            team2_rows = group[group['team'] == team2]

            # Calculate averages across sportsbooks for both teams
            team1_avg_odds = team1_rows['numeric_odds'].mean()
            team1_avg_implied_odds = team1_rows['implied_odds'].mean()
            team1_avg_wager_percentage = team1_rows['wager_percentage'].mean()

            team2_avg_odds = team2_rows['numeric_odds'].mean()
            team2_avg_implied_odds = team2_rows['implied_odds'].mean()
            team2_avg_wager_percentage = team2_rows['wager_percentage'].mean()

            # Skip if any average odds are NaN
            if pd.isna(team1_avg_odds) or pd.isna(team2_avg_odds):
                continue

            # **Calculate the bookmaker's vig**
            sum_implied_odds = team1_avg_implied_odds + team2_avg_implied_odds

            if pd.isna(sum_implied_odds) or sum_implied_odds == 0:
                continue  # Skip games with invalid implied odds

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

        # Initialize tqdm for apply
        tqdm.pandas(desc="Matching odds")

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
                    'home_odds': np.nan,
                    'away_odds': np.nan,
                    'home_implied_odds': np.nan,
                    'away_implied_odds': np.nan,
                    'home_wager_percentage': np.nan,
                    'away_wager_percentage': np.nan,
                    'vig': np.nan
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
                    'home_odds': np.nan,
                    'away_odds': np.nan,
                    'home_implied_odds': np.nan,
                    'away_implied_odds': np.nan,
                    'home_wager_percentage': np.nan,
                    'away_wager_percentage': np.nan,
                    'vig': np.nan
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

        # Apply the function to each row with progress bar
        odds_data = merged_df.progress_apply(find_matching_odds, axis=1)

        # Concatenate the odds data with the merged_df
        merged_df = pd.concat([merged_df, odds_data], axis=1)

        return merged_df

    def calculate_weighted_average(self, df: pd.DataFrame, weight_column: str) -> dict:
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
        if total_weight == 0 or pd.isna(total_weight):
            # Avoid division by zero
            for col in numeric_cols:
                weighted_avg[col] = 0
        else:
            for col in numeric_cols:
                weighted_avg[col] = (df[col] * df[weight_column]).sum() / total_weight

        return weighted_avg
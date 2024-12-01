import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# Configure logging to help debug and monitor the scraping process
logging.basicConfig(
    filename='odds_matching_engine.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


class OddsMatchingEngine:
    def __init__(
            self,
            moneylines_df,
            stat_df=None,
            player_df=None,
            min_odds=1.1,        # Decimal odds
            max_odds=10,         # Decimal odds
            iqr_multiplier=1.5
    ):
        """
        Initializes the OddsMatchingEngine with dataframes and outlier parameters.

        Parameters:
            moneylines_df (pd.DataFrame): DataFrame containing moneyline odds data.
            stat_df (pd.DataFrame, optional): DataFrame containing statistical data.
            player_df (pd.DataFrame, optional): DataFrame containing player data.
            min_odds (float, optional): Minimum decimal odds to retain. Defaults to 1.1.
            max_odds (float, optional): Maximum decimal odds to retain. Defaults to 10.
            iqr_multiplier (float, optional): Multiplier for IQR to determine statistical outliers. Defaults to 1.5.
        """
        self.moneylines_df = moneylines_df.copy()
        self.stat_df = stat_df.copy() if stat_df is not None else None
        self.player_df = player_df.copy() if player_df is not None else None
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.iqr_multiplier = iqr_multiplier

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
        if pd.isna(abbr):
            logging.warning("Encountered NaN or None in team abbreviation. Assigning as 'UNKNOWN'.")
            return 'UNKNOWN'
        standardized = abbr_mapping.get(abbr.upper(), abbr.upper())
        if standardized != abbr.upper():
            logging.debug(f"Standardized team abbreviation: {abbr} -> {standardized}")
        return standardized

    @staticmethod
    def convert_odds_to_numeric(odds):
        """
        Converts American moneyline odds from string format to integer.

        Parameters:
            odds (str or int): American moneyline odds (e.g., '-150', '+120').

        Returns:
            int or np.nan: Numeric representation of odds or NaN if invalid.
        """
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
            numeric_odds = -odds_int if negative else odds_int
            logging.debug(f"Converted odds: {odds} -> {numeric_odds}")
            return numeric_odds
        except ValueError:
            # Handle invalid odds strings
            logging.warning(f"Invalid odds format encountered: '{odds}'")
            return np.nan

    @staticmethod
    def moneyline_to_decimal(moneyline):
        """
        Convert American moneyline odds to decimal odds.

        Parameters:
            moneyline (float or array-like): American moneyline odds.

        Returns:
            float or np.ndarray: Decimal odds.
        """
        moneyline = np.asarray(moneyline, dtype=float)
        decimal_odds = np.where(
            moneyline > 0,
            (moneyline / 100) + 1,
            np.where(
                moneyline < 0,
                (100 / np.abs(moneyline)) + 1,
                1.0  # Represents no payout
            )
        )
        return decimal_odds

    @staticmethod
    def decimal_to_moneyline(decimal_odds):
        """
        Convert decimal odds to American odds.

        Parameters:
            decimal_odds (float or array-like): Decimal odds to convert.

        Returns:
            float or np.ndarray: Converted American odds. Returns NaN for invalid decimal odds.
        """
        decimal_odds = np.asarray(decimal_odds, dtype=float)
        american_odds = np.where(
            decimal_odds > 2.0,
            (decimal_odds - 1) * 100,
            np.where(
                decimal_odds < 2.0,
                -100 / (decimal_odds - 1),
                100  # When decimal_odds == 2.0
            )
        )
        # Assign NaN to invalid decimal odds (<=1.0)
        american_odds = np.where(decimal_odds <= 1.0, np.nan, american_odds)
        return american_odds

    @staticmethod
    def calculate_implied_odds(odds):
        """
        Calculates the implied probability from decimal odds.

        Parameters:
            odds (float): Decimal odds.

        Returns:
            float: Implied probability.
        """
        if pd.isna(odds):
            return np.nan
        if odds <= 1.0:
            return np.nan  # Invalid decimal odds
        return 1 / odds

    def prepare_and_calculate_odds(self):
        """
        Processes the moneylines dataframe to pair the odds for each game into a single row,
        removing outliers based on specified decimal odds thresholds and statistical measures.

        Returns:
            pd.DataFrame: DataFrame containing processed game odds with both decimal and American formats.
        """
        logging.info("Starting preparation and calculation of odds.")

        # Standardize team abbreviations
        self.moneylines_df['team'] = self.moneylines_df['team'].apply(self.standardize_team_abbr)
        self.moneylines_df['opponent'] = self.moneylines_df['opponent'].apply(self.standardize_team_abbr)

        # Drop rows where 'team' or 'opponent' is 'UNKNOWN'
        before_drop_unknown = len(self.moneylines_df)
        self.moneylines_df = self.moneylines_df[
            (self.moneylines_df['team'] != 'UNKNOWN') &
            (self.moneylines_df['opponent'] != 'UNKNOWN')
        ]
        dropped_unknown = before_drop_unknown - len(self.moneylines_df)
        if dropped_unknown > 0:
            logging.warning(f"Dropped {dropped_unknown} rows due to 'UNKNOWN' team abbreviations.")

        # Convert 'date' to datetime
        self.moneylines_df['date'] = pd.to_datetime(self.moneylines_df['date'], errors='coerce')
        initial_row_count = len(self.moneylines_df)
        self.moneylines_df.dropna(subset=['date'], inplace=True)
        if len(self.moneylines_df) < initial_row_count:
            dropped = initial_row_count - len(self.moneylines_df)
            logging.warning(f"Dropped {dropped} rows due to invalid dates.")

        # Clean and convert 'wager_percentage'
        self.moneylines_df['wager_percentage'] = self.moneylines_df['wager_percentage'].str.replace('%', '', regex=False)
        self.moneylines_df['wager_percentage'] = self.moneylines_df['wager_percentage'].replace('-', np.nan)
        self.moneylines_df['wager_percentage'] = pd.to_numeric(
            self.moneylines_df['wager_percentage'],
            errors='coerce'
        )

        # Handle NaN values by imputing with median wager percentage
        median_wager = self.moneylines_df['wager_percentage'].median()
        self.moneylines_df['wager_percentage'].fillna(median_wager, inplace=True)
        logging.info(f"Filled NaN wager percentages with median value: {median_wager}")

        # **Convert American odds to Numeric Odds**
        self.moneylines_df['numeric_odds'] = self.moneylines_df['odds'].apply(self.convert_odds_to_numeric)

        # **Convert Numeric Odds to Decimal Odds**
        self.moneylines_df['decimal_odds'] = self.moneyline_to_decimal(self.moneylines_df['numeric_odds'])

        # Drop rows with NaN 'decimal_odds'
        before_drop_na = len(self.moneylines_df)
        self.moneylines_df.dropna(subset=['decimal_odds'], inplace=True)
        dropped_na = before_drop_na - len(self.moneylines_df)
        if dropped_na > 0:
            logging.warning(f"Dropped {dropped_na} rows due to NaN decimal_odds.")

        # **Create a unique game identifier before any filtering**
        self.moneylines_df['game_id'] = self.moneylines_df.apply(
            lambda row: f"{row['date'].strftime('%Y-%m-%d')}_{'_'.join(sorted([row['team'], row['opponent']]))}",
            axis=1
        )
        logging.debug("Created 'game_id' for all rows.")

        # **Remove outliers based on decimal odds**
        # Define acceptable range in decimal odds corresponding to min_odds and max_odds
        # min_odds and max_odds are already in decimal format
        condition = self.moneylines_df['decimal_odds'].between(self.min_odds, self.max_odds, inclusive='both')
        before_filter = len(self.moneylines_df)
        self.moneylines_df = self.moneylines_df[condition]
        filtered_out = before_filter - len(self.moneylines_df)
        if filtered_out > 0:
            logging.info(
                f"Filtered out {filtered_out} rows based on decimal odds outside the range [{self.min_odds}, {self.max_odds}].")

        # **Remove statistical outliers using IQR method within each game_id and team**
        def remove_statistical_outliers(group):
            Q1 = group['decimal_odds'].quantile(0.25)
            Q3 = group['decimal_odds'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            initial_len = len(group)
            filtered_group = group[
                group['decimal_odds'].between(lower_bound, upper_bound, inclusive='both')
            ]
            outliers_removed = initial_len - len(filtered_group)
            if outliers_removed > 0:
                logging.info(
                    f"Removed {outliers_removed} statistical outliers from game_id '{group.name[0]}' and team '{group.name[1]}'.")
            return filtered_group

        # Apply statistical outlier removal per game_id and team
        self.moneylines_df = self.moneylines_df.groupby(['game_id', 'team']).apply(
            remove_statistical_outliers).reset_index(drop=True)

        # Calculate implied probabilities based on decimal odds
        self.moneylines_df['implied_probability'] = self.moneylines_df['decimal_odds'].apply(
            self.calculate_implied_odds)

        # Group by game_id
        grouped_games = self.moneylines_df.groupby('game_id')

        # Initialize list to collect game odds
        game_odds_list = []

        # Get total number of games for progress bar
        total_games = self.moneylines_df['game_id'].nunique()

        # For each game, create a single row with team1 and team2 odds
        for game_id, group in tqdm(grouped_games, total=total_games, desc='Processing games'):
            date = group['date'].iloc[0]
            teams = group['team'].unique()
            if len(teams) != 2:
                # Skip games that don't have exactly two teams
                logging.warning(
                    f"Game ID '{game_id}' skipped due to incorrect number of teams: {len(teams)} teams found.")
                continue
            team1, team2 = teams

            # Get data for both teams
            team1_rows = group[group['team'] == team1]
            team2_rows = group[group['team'] == team2]

            # Calculate averages across sportsbooks for both teams
            team1_avg_decimal_odds = team1_rows['decimal_odds'].mean()
            team1_avg_implied_prob = team1_rows['implied_probability'].mean()
            team1_avg_wager_percentage = team1_rows['wager_percentage'].mean()

            team2_avg_decimal_odds = team2_rows['decimal_odds'].mean()
            team2_avg_implied_prob = team2_rows['implied_probability'].mean()
            team2_avg_wager_percentage = team2_rows['wager_percentage'].mean()

            # Skip if any average odds are NaN
            if pd.isna(team1_avg_decimal_odds) or pd.isna(team2_avg_decimal_odds):
                logging.warning(f"Game ID '{game_id}' skipped due to NaN in average decimal odds.")
                continue

            # **Calculate the bookmaker's vig**
            sum_implied_probs = team1_avg_implied_prob + team2_avg_implied_prob

            if pd.isna(sum_implied_probs) or sum_implied_probs == 0:
                logging.warning(
                    f"Game ID '{game_id}' skipped due to invalid sum of implied probabilities: {sum_implied_probs}")
                continue  # Skip games with invalid implied probabilities

            vig = sum_implied_probs - 1

            # **Adjust implied probabilities to sum to 1**
            team1_adjusted_implied_prob = team1_avg_implied_prob / sum_implied_probs
            team2_adjusted_implied_prob = team2_avg_implied_prob / sum_implied_probs

            # **Convert average decimal odds back to American odds**
            team1_avg_moneyline_odds = self.decimal_to_moneyline(team1_avg_decimal_odds)
            team2_avg_moneyline_odds = self.decimal_to_moneyline(team2_avg_decimal_odds)

            # Create a dictionary for this game
            game_odds = {
                'game_id': game_id,
                'date': date,
                'team1': team1,
                'team2': team2,
                'team1_avg_decimal_odds': team1_avg_decimal_odds,
                'team1_avg_moneyline_odds': team1_avg_moneyline_odds,  # Converted to American odds
                'team1_avg_implied_prob': team1_adjusted_implied_prob,  # Use adjusted implied probabilities
                'team1_avg_wager_percentage': team1_avg_wager_percentage,
                'team2_avg_decimal_odds': team2_avg_decimal_odds,
                'team2_avg_moneyline_odds': team2_avg_moneyline_odds,  # Converted to American odds
                'team2_avg_implied_prob': team2_adjusted_implied_prob,  # Use adjusted implied probabilities
                'team2_avg_wager_percentage': team2_avg_wager_percentage,
                'vig': vig  # Add the vig to the game dictionary
            }

            game_odds_list.append(game_odds)

        # Convert list to DataFrame
        game_odds_df = pd.DataFrame(game_odds_list)

        # Debugging: Check if 'game_id' exists and sample data
        if 'game_id' not in game_odds_df.columns:
            logging.error("'game_id' column is missing from game_odds_df.")
        else:
            logging.debug(f"game_odds_df columns: {game_odds_df.columns.tolist()}")
            logging.debug(f"Sample game_odds_df:\n{game_odds_df.head()}")

        logging.info("Completed preparation and calculation of odds.")

        return game_odds_df

    def match_game_moneylines_pipelined(self):
        """
        Matches and appends moneyline odds, their implied probabilities, and the bookmaker's vig to the stat_df.

        Returns:
            pd.DataFrame: Merged DataFrame containing statistical data with matched odds.
        """
        logging.info("Starting matching of game moneylines to stat_df.")

        # Prepare odds and calculate implied probabilities
        game_odds_df = self.prepare_and_calculate_odds()

        if self.stat_df is None:
            logging.error("stat_df is not provided. Exiting matching process.")
            return None

        # Convert 'Game_Date' to datetime for accurate merging
        self.stat_df['Game_Date'] = pd.to_datetime(self.stat_df['Game_Date'], errors='coerce')
        before_drop_na_dates = len(self.stat_df)
        self.stat_df.dropna(subset=['Game_Date'], inplace=True)
        if len(self.stat_df) < before_drop_na_dates:
            dropped_dates = before_drop_na_dates - len(self.stat_df)
            logging.warning(f"Dropped {dropped_dates} rows from stat_df due to invalid dates.")

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
            game_id = f"{date.strftime('%Y-%m-%d')}_{'_'.join(sorted([home_team, away_team]))}"

            # Try to find matching game in game_odds_df
            odds_row = game_odds_df[game_odds_df['game_id'] == game_id]

            if odds_row.empty:
                logging.debug(f"No matching odds found for game_id: '{game_id}'")
                return pd.Series({
                    'home_odds_decimal': np.nan,
                    'home_odds_american': np.nan,
                    'away_odds_decimal': np.nan,
                    'away_odds_american': np.nan,
                    'home_implied_prob': np.nan,
                    'away_implied_prob': np.nan,
                    'home_wager_percentage': np.nan,
                    'away_wager_percentage': np.nan,
                    'vig': np.nan
                })

            odds_row = odds_row.iloc[0]

            # Determine which team is home and assign odds accordingly
            if odds_row['team1'] == home_team:
                home_decimal_odds = odds_row['team1_avg_decimal_odds']
                home_moneyline_odds = odds_row['team1_avg_moneyline_odds']
                home_implied_prob = odds_row['team1_avg_implied_prob']
                home_wager_percentage = odds_row['team1_avg_wager_percentage']
                away_decimal_odds = odds_row['team2_avg_decimal_odds']
                away_moneyline_odds = odds_row['team2_avg_moneyline_odds']
                away_implied_prob = odds_row['team2_avg_implied_prob']
                away_wager_percentage = odds_row['team2_avg_wager_percentage']
            elif odds_row['team2'] == home_team:
                home_decimal_odds = odds_row['team2_avg_decimal_odds']
                home_moneyline_odds = odds_row['team2_avg_moneyline_odds']
                home_implied_prob = odds_row['team2_avg_implied_prob']
                home_wager_percentage = odds_row['team2_avg_wager_percentage']
                away_decimal_odds = odds_row['team1_avg_decimal_odds']
                away_moneyline_odds = odds_row['team1_avg_moneyline_odds']
                away_implied_prob = odds_row['team1_avg_implied_prob']
                away_wager_percentage = odds_row['team1_avg_wager_percentage']
            else:
                # Teams do not match, cannot assign odds
                logging.warning(f"Team mismatch for game_id: '{game_id}'")
                return pd.Series({
                    'home_odds_decimal': np.nan,
                    'home_odds': np.nan,
                    'away_odds_decimal': np.nan,
                    'away_odds': np.nan,
                    'home_implied_prob': np.nan,
                    'away_implied_prob': np.nan,
                    'home_wager_percentage': np.nan,
                    'away_wager_percentage': np.nan,
                    'vig': np.nan
                })

            # Include the vig
            vig = odds_row['vig']

            return pd.Series({
                'home_odds_decimal': home_decimal_odds,
                'home_odds': home_moneyline_odds,
                'away_odds_decimal': away_decimal_odds,
                'away_odds': away_moneyline_odds,
                'home_implied_prob': home_implied_prob,
                'away_implied_prob': away_implied_prob,
                'home_wager_percentage': home_wager_percentage,
                'away_wager_percentage': away_wager_percentage,
                'vig': vig
            })

        # Apply the function to each row with progress bar
        odds_data = merged_df.progress_apply(find_matching_odds, axis=1)

        # Concatenate the odds data with the merged_df
        merged_df = pd.concat([merged_df, odds_data], axis=1)

        logging.info("Completed matching of game moneylines to stat_df.")

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
            logging.warning("Empty DataFrame or weight column not found.")
            return {}

        # Select numeric columns for calculation
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if weight_column in numeric_cols:
            numeric_cols.remove(weight_column)

        # Calculate weighted average
        weighted_avg = {}
        total_weight = df[weight_column].sum()
        if total_weight == 0 or pd.isna(total_weight):
            logging.warning("Total weight is zero or NaN. Returning zeros for weighted averages.")
            for col in numeric_cols:
                weighted_avg[col] = 0
        else:
            for col in numeric_cols:
                weighted_avg[col] = (df[col] * df[weight_column]).sum() / total_weight
                logging.debug(f"Weighted average for {col}: {weighted_avg[col]}")

        return weighted_avg

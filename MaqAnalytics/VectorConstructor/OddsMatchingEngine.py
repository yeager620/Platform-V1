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
            moneylines_df (pd.DataFrame): DataFrame containing moneyline odds data (must include 'opening_line').
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
                1.0  # Represents no payout / invalid
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
        if pd.isna(odds) or odds <= 1.0:
            return np.nan
        return 1 / odds

    def prepare_and_calculate_odds(self):
        """
        Processes the moneylines dataframe to pair the odds for each game into a single row,
        removing outliers based on specified decimal odds thresholds and statistical measures.

        Returns:
            pd.DataFrame: DataFrame containing processed game odds with both decimal and American formats,
                          including opening line columns and implied probabilities.
        """
        logging.info("Starting preparation and calculation of odds.")

        # 1) Standardize team abbreviations
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

        # 2) Convert 'date' to datetime
        self.moneylines_df['date'] = pd.to_datetime(self.moneylines_df['date'], errors='coerce')
        initial_row_count = len(self.moneylines_df)
        self.moneylines_df.dropna(subset=['date'], inplace=True)
        if len(self.moneylines_df) < initial_row_count:
            dropped = initial_row_count - len(self.moneylines_df)
            logging.warning(f"Dropped {dropped} rows due to invalid dates.")

        # 3) Clean and convert 'wager_percentage'
        self.moneylines_df['wager_percentage'] = (
            self.moneylines_df['wager_percentage'].str.replace('%', '', regex=False)
        )
        self.moneylines_df['wager_percentage'] = self.moneylines_df['wager_percentage'].replace('-', np.nan)
        self.moneylines_df['wager_percentage'] = pd.to_numeric(
            self.moneylines_df['wager_percentage'],
            errors='coerce'
        )

        # Handle NaN values by imputing with median wager percentage
        median_wager = self.moneylines_df['wager_percentage'].median()
        self.moneylines_df['wager_percentage'].fillna(median_wager, inplace=True)
        logging.info(f"Filled NaN wager percentages with median value: {median_wager}")

        # -------------------------- Main Odds Conversion ---------------------------
        # 4) Convert American 'odds' (current odds) to numeric
        self.moneylines_df['numeric_odds'] = self.moneylines_df['odds'].apply(self.convert_odds_to_numeric)
        # 5) Convert numeric to decimal
        self.moneylines_df['decimal_odds'] = self.moneyline_to_decimal(self.moneylines_df['numeric_odds'])
        # 5b) Calculate implied probability for current odds
        self.moneylines_df['implied_probability'] = self.moneylines_df['decimal_odds'].apply(
            self.calculate_implied_odds
        )

        # Drop rows with NaN 'decimal_odds'
        before_drop_na = len(self.moneylines_df)
        self.moneylines_df.dropna(subset=['decimal_odds'], inplace=True)
        dropped_na = before_drop_na - len(self.moneylines_df)
        if dropped_na > 0:
            logging.warning(f"Dropped {dropped_na} rows due to NaN decimal_odds.")

        # ------------------------- Opening Line Conversion --------------------------
        # Make sure 'opening_line' column exists; if not, create a placeholder
        if 'opening_line' not in self.moneylines_df.columns:
            logging.warning("'opening_line' column not found in moneylines_df. Creating empty column.")
            self.moneylines_df['opening_line'] = np.nan

        # Convert 'opening_line' to American numeric
        self.moneylines_df['opening_numeric_odds'] = self.moneylines_df['opening_line'].apply(
            self.convert_odds_to_numeric
        )
        # Convert to decimal
        self.moneylines_df['opening_decimal_odds'] = self.moneyline_to_decimal(
            self.moneylines_df['opening_numeric_odds']
        )
        # Calculate implied probability for opening lines
        self.moneylines_df['opening_implied_probability'] = self.moneylines_df['opening_decimal_odds'].apply(
            self.calculate_implied_odds
        )
        # We won't drop rows if opening_decimal_odds is NaN.

        # 6) Create a unique game identifier
        self.moneylines_df['game_id'] = self.moneylines_df.apply(
            lambda row: f"{row['date'].strftime('%Y-%m-%d')}_{'_'.join(sorted([row['team'], row['opponent']]))}",
            axis=1
        )
        logging.debug("Created 'game_id' for all rows.")

        # 7) Remove outliers for main 'decimal_odds' based on min/max
        condition = self.moneylines_df['decimal_odds'].between(self.min_odds, self.max_odds, inclusive='both')
        before_filter = len(self.moneylines_df)
        self.moneylines_df = self.moneylines_df[condition]
        filtered_out = before_filter - len(self.moneylines_df)
        if filtered_out > 0:
            logging.info(
                f"Filtered out {filtered_out} rows based on decimal_odds outside the range [{self.min_odds}, {self.max_odds}].")

        # 8) (Optional) Apply the same min/max filtering to opening_decimal_odds
        condition_opening = (
            self.moneylines_df['opening_decimal_odds'].between(self.min_odds, self.max_odds, inclusive='both')
            | self.moneylines_df['opening_decimal_odds'].isna()
        )
        before_filter_opening = len(self.moneylines_df)
        self.moneylines_df = self.moneylines_df[condition_opening]
        filtered_opening_out = before_filter_opening - len(self.moneylines_df)
        if filtered_opening_out > 0:
            logging.info(
                f"Filtered out {filtered_opening_out} rows based on opening_decimal_odds outside [{self.min_odds}, {self.max_odds}].")

        # 9) Remove statistical outliers for decimal_odds and opening_decimal_odds
        def remove_statistical_outliers(group):
            # For main odds
            Q1_main = group['decimal_odds'].quantile(0.25)
            Q3_main = group['decimal_odds'].quantile(0.75)
            IQR_main = Q3_main - Q1_main
            lower_main = Q1_main - self.iqr_multiplier * IQR_main
            upper_main = Q3_main + self.iqr_multiplier * IQR_main

            # For opening odds
            Q1_open = group['opening_decimal_odds'].quantile(0.25)
            Q3_open = group['opening_decimal_odds'].quantile(0.75)
            IQR_open = Q3_open - Q1_open
            lower_open = Q1_open - self.iqr_multiplier * IQR_open
            upper_open = Q3_open + self.iqr_multiplier * IQR_open

            initial_len = len(group)
            filtered_group = group[
                group['decimal_odds'].between(lower_main, upper_main, inclusive='both') &
                (
                    group['opening_decimal_odds'].between(lower_open, upper_open, inclusive='both')
                    | group['opening_decimal_odds'].isna()
                )
            ]
            outliers_removed = initial_len - len(filtered_group)
            if outliers_removed > 0:
                logging.info(
                    f"Removed {outliers_removed} outliers in game_id '{group.name}'."
                )
            return filtered_group

        self.moneylines_df = self.moneylines_df.groupby('game_id').apply(remove_statistical_outliers).reset_index(drop=True)

        # -------------------------------------------------------------------------
        # 10) Group by game_id to compute averages
        # -------------------------------------------------------------------------
        grouped_games = self.moneylines_df.groupby('game_id')
        game_odds_list = []
        total_games = self.moneylines_df['game_id'].nunique()

        for game_id, group in tqdm(grouped_games, total=total_games, desc='Processing games'):
            date = group['date'].iloc[0]
            teams = group['team'].unique()
            if len(teams) != 2:
                # Skip if not exactly two teams
                logging.warning(
                    f"Game ID '{game_id}' skipped due to incorrect number of teams: {len(teams)}."
                )
                continue
            team1, team2 = teams

            # Averages for Team 1
            team1_rows = group[group['team'] == team1]
            team1_avg_decimal_odds = team1_rows['decimal_odds'].mean()
            team1_avg_implied_prob = team1_rows['implied_probability'].mean()
            team1_avg_wager_percentage = team1_rows['wager_percentage'].mean()

            # Opening lines for Team 1
            team1_avg_opening_decimal_odds = team1_rows['opening_decimal_odds'].mean()
            team1_avg_opening_implied_prob = team1_rows['opening_implied_probability'].mean()

            # Convert decimal to American
            team1_avg_moneyline_odds = self.decimal_to_moneyline(team1_avg_decimal_odds)
            team1_avg_opening_moneyline = self.decimal_to_moneyline(team1_avg_opening_decimal_odds)

            # Averages for Team 2
            team2_rows = group[group['team'] == team2]
            team2_avg_decimal_odds = team2_rows['decimal_odds'].mean()
            team2_avg_implied_prob = team2_rows['implied_probability'].mean()
            team2_avg_wager_percentage = team2_rows['wager_percentage'].mean()

            # Opening lines for Team 2
            team2_avg_opening_decimal_odds = team2_rows['opening_decimal_odds'].mean()
            team2_avg_opening_implied_prob = team2_rows['opening_implied_probability'].mean()

            team2_avg_moneyline_odds = self.decimal_to_moneyline(team2_avg_decimal_odds)
            team2_avg_opening_moneyline = self.decimal_to_moneyline(team2_avg_opening_decimal_odds)

            # Skip if any average odds are NaN
            if pd.isna(team1_avg_decimal_odds) or pd.isna(team2_avg_decimal_odds):
                logging.warning(f"Game ID '{game_id}' skipped due to NaN in average decimal odds.")
                continue

            # Calculate vig for **current** lines
            sum_implied_probs = (team1_avg_implied_prob + team2_avg_implied_prob)
            if pd.isna(sum_implied_probs) or sum_implied_probs == 0:
                logging.warning(
                    f"Game ID '{game_id}' skipped due to invalid sum of implied probabilities: {sum_implied_probs}"
                )
                continue
            vig = sum_implied_probs - 1

            # Adjust implied probabilities so they sum to 1
            team1_adjusted_implied_prob = team1_avg_implied_prob / sum_implied_probs
            team2_adjusted_implied_prob = team2_avg_implied_prob / sum_implied_probs

            # --------------------------
            # Opening line implied probs
            # --------------------------
            # We can similarly calculate an 'opening vig' for each game
            if not pd.isna(team1_avg_opening_implied_prob) and not pd.isna(team2_avg_opening_implied_prob):
                sum_opening_implied_probs = team1_avg_opening_implied_prob + team2_avg_opening_implied_prob
                opening_vig = sum_opening_implied_probs - 1
                # Adjust opening implied probabilities
                # Avoid dividing by zero if sum_opening_implied_probs <= 0
                if sum_opening_implied_probs > 0:
                    team1_opening_adjusted_prob = team1_avg_opening_implied_prob / sum_opening_implied_probs
                    team2_opening_adjusted_prob = team2_avg_opening_implied_prob / sum_opening_implied_probs
                else:
                    team1_opening_adjusted_prob = np.nan
                    team2_opening_adjusted_prob = np.nan
                    opening_vig = np.nan
            else:
                # If either opening line is NaN, set these to NaN
                team1_opening_adjusted_prob = np.nan
                team2_opening_adjusted_prob = np.nan
                opening_vig = np.nan

            game_odds = {
                'game_id': game_id,
                'date': date,
                'team1': team1,
                'team2': team2,

                # Current lines
                'team1_avg_decimal_odds': team1_avg_decimal_odds,
                'team1_avg_moneyline_odds': team1_avg_moneyline_odds,
                'team1_avg_implied_prob': team1_adjusted_implied_prob,
                'team1_avg_wager_percentage': team1_avg_wager_percentage,

                'team2_avg_decimal_odds': team2_avg_decimal_odds,
                'team2_avg_moneyline_odds': team2_avg_moneyline_odds,
                'team2_avg_implied_prob': team2_adjusted_implied_prob,
                'team2_avg_wager_percentage': team2_avg_wager_percentage,

                'vig': vig,

                # Opening lines
                'team1_avg_opening_decimal_odds': team1_avg_opening_decimal_odds,
                'team1_avg_opening_moneyline': team1_avg_opening_moneyline,
                'team1_avg_opening_implied_prob': team1_opening_adjusted_prob,

                'team2_avg_opening_decimal_odds': team2_avg_opening_decimal_odds,
                'team2_avg_opening_moneyline': team2_avg_opening_moneyline,
                'team2_avg_opening_implied_prob': team2_opening_adjusted_prob,

                # Additional opening vig if you want it
                'opening_vig': opening_vig
            }
            game_odds_list.append(game_odds)

        game_odds_df = pd.DataFrame(game_odds_list)

        if 'game_id' not in game_odds_df.columns:
            logging.error("'game_id' column is missing from game_odds_df.")
        else:
            logging.debug(f"game_odds_df columns: {game_odds_df.columns.tolist()}")
            logging.debug(f"Sample game_odds_df:\n{game_odds_df.head()}")

        logging.info("Completed preparation and calculation of odds.")

        return game_odds_df

    def match_game_moneylines_pipelined(self):
        """
        Matches and appends moneyline odds (including opening lines + implied probabilities) to the stat_df.

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

        merged_df = self.stat_df.copy()

        tqdm.pandas(desc="Matching odds")

        # Helper function to match each row in stat_df to the aggregated game_odds
        def find_matching_odds(row):
            date = row['Game_Date']
            home_team = row['Home_Team_Abbr']
            away_team = row['Away_Team_Abbr']

            # Create game_id
            game_id = f"{date.strftime('%Y-%m-%d')}_{'_'.join(sorted([home_team, away_team]))}"
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
                    'vig': np.nan,

                    # Opening lines
                    'home_opening_odds_decimal': np.nan,
                    'home_opening_odds_american': np.nan,
                    'away_opening_odds_decimal': np.nan,
                    'away_opening_odds_american': np.nan,
                    'home_opening_implied_prob': np.nan,
                    'away_opening_implied_prob': np.nan,
                    'opening_vig': np.nan
                })

            odds_row = odds_row.iloc[0]

            # Determine which side is home vs away
            if odds_row['team1'] == home_team:
                home_decimal_odds = odds_row['team1_avg_decimal_odds']
                home_moneyline_odds = odds_row['team1_avg_moneyline_odds']
                home_implied_prob = odds_row['team1_avg_implied_prob']
                home_wager_percentage = odds_row['team1_avg_wager_percentage']

                home_opening_decimal = odds_row['team1_avg_opening_decimal_odds']
                home_opening_moneyline = odds_row['team1_avg_opening_moneyline']
                home_opening_implied_prob = odds_row['team1_avg_opening_implied_prob']

                away_decimal_odds = odds_row['team2_avg_decimal_odds']
                away_moneyline_odds = odds_row['team2_avg_moneyline_odds']
                away_implied_prob = odds_row['team2_avg_implied_prob']
                away_wager_percentage = odds_row['team2_avg_wager_percentage']

                away_opening_decimal = odds_row['team2_avg_opening_decimal_odds']
                away_opening_moneyline = odds_row['team2_avg_opening_moneyline']
                away_opening_implied_prob = odds_row['team2_avg_opening_implied_prob']

            elif odds_row['team2'] == home_team:
                home_decimal_odds = odds_row['team2_avg_decimal_odds']
                home_moneyline_odds = odds_row['team2_avg_moneyline_odds']
                home_implied_prob = odds_row['team2_avg_implied_prob']
                home_wager_percentage = odds_row['team2_avg_wager_percentage']

                home_opening_decimal = odds_row['team2_avg_opening_decimal_odds']
                home_opening_moneyline = odds_row['team2_avg_opening_moneyline']
                home_opening_implied_prob = odds_row['team2_avg_opening_implied_prob']

                away_decimal_odds = odds_row['team1_avg_decimal_odds']
                away_moneyline_odds = odds_row['team1_avg_moneyline_odds']
                away_implied_prob = odds_row['team1_avg_implied_prob']
                away_wager_percentage = odds_row['team1_avg_wager_percentage']

                away_opening_decimal = odds_row['team1_avg_opening_decimal_odds']
                away_opening_moneyline = odds_row['team1_avg_opening_moneyline']
                away_opening_implied_prob = odds_row['team1_avg_opening_implied_prob']
            else:
                # Teams do not match
                logging.warning(f"Team mismatch for game_id: '{game_id}'")
                return pd.Series({
                    'home_odds_decimal': np.nan,
                    'home_odds_american': np.nan,
                    'away_odds_decimal': np.nan,
                    'away_odds_american': np.nan,
                    'home_implied_prob': np.nan,
                    'away_implied_prob': np.nan,
                    'home_wager_percentage': np.nan,
                    'away_wager_percentage': np.nan,
                    'vig': np.nan,

                    'home_opening_odds_decimal': np.nan,
                    'home_opening_odds_american': np.nan,
                    'away_opening_odds_decimal': np.nan,
                    'away_opening_odds_american': np.nan,
                    'home_opening_implied_prob': np.nan,
                    'away_opening_implied_prob': np.nan,
                    'opening_vig': np.nan
                })

            vig = odds_row['vig']
            opening_vig = odds_row['opening_vig']

            return pd.Series({
                'home_odds_decimal': home_decimal_odds,
                'home_odds_american': home_moneyline_odds,
                'away_odds_decimal': away_decimal_odds,
                'away_odds_american': away_moneyline_odds,
                'home_implied_prob': home_implied_prob,
                'away_implied_prob': away_implied_prob,
                'home_wager_percentage': home_wager_percentage,
                'away_wager_percentage': away_wager_percentage,
                'vig': vig,

                # Opening lines
                'home_opening_odds_decimal': home_opening_decimal,
                'home_opening_odds_american': home_opening_moneyline,
                'away_opening_odds_decimal': away_opening_decimal,
                'away_opening_odds_american': away_opening_moneyline,

                'home_opening_implied_prob': home_opening_implied_prob,
                'away_opening_implied_prob': away_opening_implied_prob,
                'opening_vig': opening_vig
            })

        odds_data = merged_df.progress_apply(find_matching_odds, axis=1)
        merged_df = pd.concat([merged_df, odds_data], axis=1)

        logging.info("Completed matching of game moneylines (including opening lines + implied probabilities) to stat_df.")
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

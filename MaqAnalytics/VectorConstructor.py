import pandas as pd
import asyncio
from datetime import datetime, timedelta
from SportsBookReview import SportsbookReviewScraper
from bs_retrosheet_converter import SavantRetrosheetConverter


class VectorConstructor:
    def __init__(self, player_df, sportsbook_scraper: SportsbookReviewScraper, moneylines_df=None):
        """
        Initializes the VectorConstructor with necessary components.

        Parameters:
            player_df (pd.DataFrame): DataFrame containing player statistics.
            sportsbook_scraper (SportsbookReviewScraper): Instance for scraping SportsBookReview odds data.
            moneylines_df (pd.DataFrame): DataFrame containing historical moneyline data. If None, it will be fetched using the scraper.
        """
        self.player_df = player_df
        self.sportsbook_scraper = sportsbook_scraper
        self.moneylines_df = moneylines_df or self.scrape_moneylines()
        self.abb_dict = {
            "ARI": "ARI",  # Arizona Diamondbacks
            "ATL": "ATL",  # Atlanta Braves
            "BAL": "BAL",  # Baltimore Orioles
            "BOS": "BOS",  # Boston Red Sox
            "CHC": "CHN",  # Chicago Cubs
            "CWS": "CHA",  # Chicago White Sox
            "CIN": "CIN",  # Cincinnati Reds
            "CLE": "CLE",  # Cleveland Guardians
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
        home_team_abbr = game_json['scoreboard']['teams']['home']['team']['abbreviation']
        away_team_abbr = game_json['scoreboard']['teams']['away']['team']['abbreviation']

        return game_date, home_team_abbr, away_team_abbr

    async def fetch_game_odds(self, game_json):
        """
        Fetches moneyline odds for a specific game from SportsBookReview.

        Parameters:
            game_json (dict): JSON data for the game.

        Returns:
            pd.DataFrame: DataFrame containing moneyline odds for the game.
        """
        # Parse game details
        game_date, home_team, away_team = self.parse_game_details(game_json)

        # Scrape odds data for the game date
        odds_df = self.sportsbook_scraper.scrape()

        # Filter odds for the specific game based on teams and date
        game_odds = odds_df[
            (odds_df['date'] == game_date) &
            (
                    (
                            self.abb_dict[odds_df['team']] == home_team
                    ) &
                    (
                            self.abb_dict[odds_df['opponent']] == away_team
                    )
            )
            ]

        return game_odds

    def calculate_average_moneyline(self, moneylines_df):
        """
        Calculates the average moneyline odds for home and away teams.

        Parameters:
            moneylines_df (pd.DataFrame): DataFrame containing moneyline data for a specific game.

        Returns:
            pd.DataFrame: DataFrame with average moneyline odds and implied probabilities.
        """
        # Convert price column to numeric, ignoring errors for non-numeric values
        moneylines_df['odds'] = pd.to_numeric(moneylines_df['odds'], errors='coerce')

        # Drop rows with NaN odds
        moneylines_df = moneylines_df.dropna(subset=['odds'])

        # Calculate average odds for home and away teams
        average_odds = moneylines_df.groupby(['team', 'opponent'])['odds'].mean().reset_index()

        # Calculate implied probabilities
        average_odds['home_team_implied_odds'] = average_odds['odds'].apply(self.calculate_implied_odds)
        average_odds['away_team_implied_odds'] = average_odds['odds'].apply(self.calculate_implied_odds)

        return average_odds

    @staticmethod
    def calculate_implied_odds(moneyline):
        """
        Calculates the implied probability from moneyline odds.

        Parameters:
            moneyline (float): Moneyline odds.

        Returns:
            float: Implied probability.
        """
        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return -moneyline / (-moneyline + 100)

    @staticmethod
    def extract_team_players(game_json):
        """
        Extracts player IDs for home and away teams, segregating batters and pitchers.

        Parameters:
            game_json (dict): JSON data for the game.

        Returns:
            dict: Dictionary containing lists of home batters, home pitchers, away batters, away pitchers.
        """
        home_batters = []
        home_pitchers = []
        away_batters = []
        away_pitchers = []

        players = game_json['boxscore']['teams']['home'].get('players', {})
        for player_key, player_info in players.items():
            position = player_info.get('position', {}).get('code', '')
            player_id = player_info.get('person', {}).get('id')
            if position == 'P':
                home_pitchers.append(player_id)
            else:
                home_batters.append(player_id)

        players = game_json['boxscore']['teams']['away'].get('players', {})
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
        home_team_score = game_json['scoreboard']['teams']['home']['score']
        away_team_score = game_json['scoreboard']['teams']['away']['score']
        feature_vector['home_team_won'] = 1 if home_team_score > away_team_score else 0

        return feature_vector

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
        if total_weight == 0:
            # Avoid division by zero
            for col in numeric_cols:
                weighted_avg[col] = 0
        else:
            for col in numeric_cols:
                weighted_avg[col] = (df[col] * df[weight_column]).sum() / total_weight

        return weighted_avg

    def scrape_moneylines(self):
        """
        Scrapes historical moneyline data using the SportsbookReviewScraper.

        Returns:
            pd.DataFrame: DataFrame containing historical moneyline data.
        """
        print("Scraping historical moneyline data from SportsBookReview...")
        moneylines_df = self.sportsbook_scraper.scrape()
        print("Moneyline data scraping completed.")
        return moneylines_df

    async def construct_all_game_vectors(self, date_ranges=None):
        """
        Constructs feature vectors for all games within the specified date ranges.

        Parameters:
            date_ranges (list of tuples): Each tuple contains (start_date, end_date) in "MM/DD/YYYY" format.

        Returns:
            pd.DataFrame: DataFrame containing feature vectors and target variables for all games.
        """
        # Initialize the game logs fetching class
        retrosheet_df = self.savant_converter.process_games_retrosheet_with_outcome()

        # Iterate over each game in retrosheet_df and construct feature vectors
        feature_vectors = []
        for _, row in retrosheet_df.iterrows():
            game_id = row['game_id']
            game_json = self.savant_converter.fetch_gamelog(game_id)
            if not game_json:
                print(f"Game JSON for game_id {game_id} not found. Skipping.")
                continue

            # Extract player IDs and fetch their stats up to the game date
            player_ids = self.extract_team_players(game_json)
            game_date = row['date']
            player_data_timeframe = datetime.strptime(game_date, '%Y-%m-%d') - timedelta(days=365)  # 1 year before game
            home_batters_df = self.fetch_player_stats(player_data_timeframe, game_date, player_ids['home_batters'])
            home_pitcher_df = self.fetch_player_stats(player_data_timeframe, game_date, player_ids['home_pitchers'])
            away_batters_df = self.fetch_player_stats(player_data_timeframe, game_date, player_ids['away_batters'])
            away_pitcher_df = self.fetch_player_stats(player_data_timeframe, game_date, player_ids['away_pitchers'])

            # Fetch and integrate moneyline data
            moneylines_df = await self.fetch_game_odds(game_json)
            moneylines_df = self.calculate_average_moneyline(moneylines_df)

            # Construct game vector
            game_vector = self.construct_game_vector(
                game_json=game_json,
                home_batters_df=home_batters_df,
                home_pitcher_df=home_pitcher_df,
                away_batters_df=away_batters_df,
                away_pitcher_df=away_pitcher_df,
                moneylines_df=moneylines_df
            )

            if game_vector:
                feature_vectors.append(game_vector)

        # Combine all feature vectors into a single DataFrame
        feature_df = pd.DataFrame(feature_vectors)

        return feature_df

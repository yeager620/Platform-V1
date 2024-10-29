import pandas as pd
import asyncio

from OddsBlaze import OddsBlazeAPI
from VectorConstructor import VectorConstructor
from bs_retrosheet_converter import SavantRetrosheetConverter


class DataPipeline:
    def __init__(
            self,
            player_df: pd.DataFrame,
            start_date: str,
            odds_api: OddsBlazeAPI = OddsBlazeAPI(),
            moneylines_df: pd.DataFrame = None
    ):
        """
        Initializes the DataPipeline with necessary components.

        Parameters:
            player_df (pd.DataFrame): DataFrame containing player statistics.
            start_date (str): Start date for fetching game logs in "MM/DD/YYYY" format.
            odds_api (OddsBlazeAPI): Instance of OddsBlazeAPI for fetching odds data.
            moneylines_df (pd.DataFrame): DataFrame containing historical moneyline data.
        """
        self.savant_converter = SavantRetrosheetConverter(start_date=start_date)
        self.vector_constructor = VectorConstructor(
            player_df=player_df,
            odds_api=odds_api,
            moneylines_df=moneylines_df
        )

    def process_games(self, date_ranges=None) -> pd.DataFrame:
        """
        Processes all games within the specified date ranges and constructs feature vectors.

        Parameters:
            date_ranges (list of tuples): Each tuple contains (start_date, end_date) in "MM/DD/YYYY" format.

        Returns:
            pd.DataFrame: DataFrame containing feature vectors and target variables for all games.
        """
        # Step 1: Fetch and process game logs using SavantRetrosheetConverter
        retrosheet_df = self.savant_converter.process_games_retrosheet_with_outcome()

        # Step 2: Fetch and process moneyline data using VectorConstructor
        # Assuming moneylines_df is already provided or fetched elsewhere
        # If not, integrate fetching moneyline data here

        # Step 3: Merge Retrosheet data with moneyline data
        # This requires that both DataFrames have matching identifiers (e.g., game_id)
        # For simplicity, let's assume 'game_id' is the key
        merged_df = pd.merge(
            retrosheet_df,
            self.vector_constructor.moneylines_df,
            on='game_id',
            how='left'
        )

        # Step 4: Construct feature vectors
        # Assuming construct_game_vector processes a row and returns features
        feature_vectors = []
        for _, row in merged_df.iterrows():
            game_json = self.savant_converter.fetch_gamelog(row['game_id'])
            if not game_json:
                continue  # Skip if game log is unavailable

            # Extract player IDs and fetch their stats up to the game date
            player_ids = self.vector_constructor.extract_team_players(game_json)
            game_date = row['date']
            player_data_timeframe = datetime.strptime(game_date, '%Y-%m-%d') - timedelta(
                days=365)  # Example: 1 year before game
            home_batters_df = self.vector_constructor.fetch_player_stats(player_data_timeframe, game_date,
                                                                         player_ids['home_batters'])
            home_pitcher_df = self.vector_constructor.fetch_player_stats(player_data_timeframe, game_date,
                                                                         player_ids['home_pitchers'])
            away_batters_df = self.vector_constructor.fetch_player_stats(player_data_timeframe, game_date,
                                                                         player_ids['away_batters'])
            away_pitcher_df = self.vector_constructor.fetch_player_stats(player_data_timeframe, game_date,
                                                                         player_ids['away_pitchers'])

            # Fetch and integrate moneyline data
            moneylines_df = self.vector_constructor.integrate_moneylines_and_odds(game_json)

            # Construct game vector
            game_vector = self.vector_constructor.construct_game_vector(
                game_json=game_json,
                home_batters_df=home_batters_df,
                home_pitcher_df=home_pitcher_df,
                away_batters_df=away_batters_df,
                away_pitcher_df=away_pitcher_df,
                moneylines_df=moneylines_df
            )

            feature_vectors.append(game_vector)

        # Combine all feature vectors into a single DataFrame
        feature_df = pd.DataFrame(feature_vectors)

        # Optional: Merge with target variables if not already included
        # This depends on how 'Home_Win' is represented in retrosheet_df and merged_df

        return feature_df

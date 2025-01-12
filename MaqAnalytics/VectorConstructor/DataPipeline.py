import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import logging
from MsgCore.SportsBookReview.SportsBookReview import SportsbookReviewScraper
from .OddsMatchingEngine import OddsMatchingEngine
from .SavantVectorGenerator import SavantVectorGenerator
from .SavantVectorGeneratorV2 import SavantVectorGeneratorV2
from .SavantVectorGeneratorV3 import SavantVectorGeneratorV3

# Configure logging to help debug and monitor the scraping process
logging.basicConfig(
    filename='data_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


class DataPipeline:
    def __init__(
            self,
            start_date: str,
            end_date: str,
            version: int = 1,
    ):
        """
        Initializes the DataPipeline with necessary components.

        Parameters:
            start_date (str): Start date for fetching game logs in "YYYY-MM-DD" format.
            end_date (str): End date for fetching game logs in "YYYY-MM-DD" format.
            version (int): Version of the SavantVectorGenerator to use (1 or 2).
        """
        self.start_date = start_date
        self.end_date = end_date

        if version == 1:
            self.savant_converter = SavantVectorGenerator(start_date=start_date, end_date=end_date)
        elif version == 2:
            self.savant_converter = SavantVectorGeneratorV2(start_date=start_date, end_date=end_date)
        elif version == 3:
            self.savant_converter = SavantVectorGeneratorV3(start_date=start_date, end_date=end_date)
        else:
            self.savant_converter = SavantVectorGenerator(start_date=start_date, end_date=end_date)

        self.moneylines_scraper = SportsbookReviewScraper(start_date=start_date, end_date=end_date)

    def process_games(self) -> pd.DataFrame:
        """
        Processes all games within the specified date ranges and constructs feature vectors.

        Returns:
            pd.DataFrame: DataFrame containing feature vectors and target variables for all games.
        """
        logging.info(f"Starting data pipeline for games from {self.start_date} to {self.end_date}.")

        # 1) Fetch and process game logs using the savant converter
        logging.info("Fetching and processing retrosheet game logs.")
        retrosheet_df = self.savant_converter.process_games_retrosheet_with_outcome()

        # 2) Fetch moneyline odds using SportsbookReviewScraper
        logging.info("Scraping moneyline odds.")
        moneylines_df = self.moneylines_scraper.scrape()

        # 3) Initialize OddsMatchingEngine and match moneylines with game stats
        logging.info("Initializing OddsMatchingEngine.")
        odds_matcher = OddsMatchingEngine(
            moneylines_df=moneylines_df,
            stat_df=retrosheet_df
        )
        logging.info("Matching moneylines with game statistics.")
        combined_df = odds_matcher.match_game_moneylines_pipelined()

        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 3):
            print(combined_df.head())

        # ------------------------------------------------------------------------------
        # Handle both column naming conventions for odds:
        #   1. (home_odds, away_odds)
        #   2. (home_odds_american, away_odds_american)
        # We unify whichever exists to (home_odds, away_odds).
        # ------------------------------------------------------------------------------
        column_set = set(combined_df.columns)

        if {'home_odds', 'away_odds'}.issubset(column_set):
            # Standard columns already exist; do nothing
            pass
        elif {'home_odds_american', 'away_odds_american'}.issubset(column_set):
            # Rename American columns to standard columns
            combined_df.rename(
                columns={
                    'home_odds_american': 'home_odds',
                    'away_odds_american': 'away_odds'
                },
                inplace=True
            )
        else:
            # Neither set of columns is available
            logging.error(
                "Could not find either ('home_odds', 'away_odds') "
                "OR ('home_odds_american', 'away_odds_american') in the DataFrame."
            )
            raise KeyError(
                "Missing both sets of odds columns: "
                "could not find 'home_odds'/'away_odds' or 'home_odds_american'/'away_odds_american'."
            )

        # Now that columns are guaranteed to be named 'home_odds' and 'away_odds':
        # Exclude rows with NaN in those columns
        initial_combined_count = len(combined_df)
        combined_df.dropna(subset=['home_odds', 'away_odds'], inplace=True)
        final_combined_count = len(combined_df)
        excluded_rows = initial_combined_count - final_combined_count
        if excluded_rows > 0:
            logging.info(f"Excluded {excluded_rows} rows with NaN in 'home_odds' or 'away_odds'.")

        # ------------------------------------------------------------------------------
        # Exclude rows where 'Home_P_numberOfPitches' or 'Away_P_numberOfPitches' is 0
        # ------------------------------------------------------------------------------
        pitching_columns = ['Home_P_numberOfPitches', 'Away_P_numberOfPitches']
        missing_pitching_columns = [col for col in pitching_columns if col not in combined_df.columns]
        if missing_pitching_columns:
            logging.error(f"Missing pitching columns in the DataFrame: {missing_pitching_columns}")
            raise KeyError(f"Missing pitching columns: {missing_pitching_columns}")

        initial_pitching_count = len(combined_df)
        pitching_exclusion = (
            (combined_df['Home_P_numberOfPitches'] == 0) |
            (combined_df['Away_P_numberOfPitches'] == 0)
        )
        combined_df = combined_df[~pitching_exclusion]
        final_pitching_count = len(combined_df)
        excluded_pitching_rows = initial_pitching_count - final_pitching_count
        if excluded_pitching_rows > 0:
            logging.info(
                f"Excluded {excluded_pitching_rows} rows where pitches = 0."
            )

        logging.info(f"Data pipeline completed. Final dataset contains {len(combined_df)} games.")
        return combined_df

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import logging
from MsgCore.SportsBookReview.SportsBookReview import SportsbookReviewScraper
from .OddsMatchingEngine import OddsMatchingEngine
from .SavantRetrosheetConverter import SavantRetrosheetConverter

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
    ):
        """
        Initializes the DataPipeline with necessary components.

        Parameters:
            start_date (str): Start date for fetching game logs in "MM/DD/YYYY" format.
            end_date (str): End date for fetching game logs in "MM/DD/YYYY" format.
        """
        self.start_date = start_date
        self.end_date = end_date

        self.savant_converter = SavantRetrosheetConverter(start_date=start_date, end_date=end_date)
        self.moneylines_scraper = SportsbookReviewScraper(start_date=start_date, end_date=end_date)

    def process_games(self) -> pd.DataFrame:
        """
        Processes all games within the specified date ranges and constructs feature vectors.

        Returns:
            pd.DataFrame: DataFrame containing feature vectors and target variables for all games.
        """
        logging.info(f"Starting data pipeline for games from {self.start_date} to {self.end_date}.")

        # Fetch and process game logs using SavantRetrosheetConverter
        logging.info("Fetching and processing retrosheet game logs.")
        retrosheet_df = self.savant_converter.process_games_retrosheet_with_outcome()

        # Fetch moneyline odds using SportsbookReviewScraper
        logging.info("Scraping moneyline odds.")
        moneylines_df = self.moneylines_scraper.scrape()

        # Initialize OddsMatchingEngine with the scraped data
        logging.info("Initializing OddsMatchingEngine.")
        odds_matcher = OddsMatchingEngine(
            moneylines_df=moneylines_df,
            stat_df=retrosheet_df
        )

        # Perform the matching of moneylines with game statistics
        logging.info("Matching moneylines with game statistics.")
        combined_df = odds_matcher.match_game_moneylines_pipelined()

        # **Exclude rows with NaN values in 'home_odds' or 'away_odds'**
        initial_combined_count = len(combined_df)
        combined_df = combined_df.dropna(subset=['home_odds', 'away_odds'])
        final_combined_count = len(combined_df)
        excluded_rows = initial_combined_count - final_combined_count
        if excluded_rows > 0:
            logging.info(f"Excluded {excluded_rows} rows with NaN 'home_odds' or 'away_odds'.")

        # **Exclude rows where 'Home_P_numberOfPitches' or 'Away_P_numberOfPitches' are 0**
        pitching_columns = ['Home_P_numberOfPitches', 'Away_P_numberOfPitches']
        missing_pitching_columns = [col for col in pitching_columns if col not in combined_df.columns]

        if missing_pitching_columns:
            logging.error(f"Missing pitching columns in the DataFrame: {missing_pitching_columns}")
            raise KeyError(f"Missing pitching columns: {missing_pitching_columns}")

        initial_pitching_count = len(combined_df)
        # Exclude rows where either Home_P_numberOfPitches or Away_P_numberOfPitches is 0
        pitching_exclusion = (combined_df['Home_P_numberOfPitches'] == 0) | (combined_df['Away_P_numberOfPitches'] == 0)
        combined_df = combined_df[~pitching_exclusion]
        final_pitching_count = len(combined_df)
        excluded_pitching_rows = initial_pitching_count - final_pitching_count
        if excluded_pitching_rows > 0:
            logging.info(f"Excluded {excluded_pitching_rows} rows with 'Home_P_numberOfPitches' or 'Away_P_numberOfPitches' equal to 0.")

        logging.info(f"Data pipeline completed. Final dataset contains {len(combined_df)} games.")

        return combined_df

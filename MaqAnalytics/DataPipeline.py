from datetime import datetime, timedelta

import pandas as pd
# import asyncio

# from OddsBlaze import OddsBlazeAPI
from MsgCore.SportsBookReview.SportsBookReview import SportsbookReviewScraper
from MaqAnalytics.VectorConstructor import VectorConstructor
from MsgCore.BaseballSavant.bs_retrosheet_converter import SavantRetrosheetConverter


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
        # date_range = (self.start_date, self.end_date)

        # Fetch and process game logs using SavantRetrosheetConverter
        retrosheet_df = self.savant_converter.process_games_retrosheet_with_outcome()

        moneylines_df = self.moneylines_scraper.scrape()

        vector_constructor = VectorConstructor(
            moneylines_df=moneylines_df,
            stat_df=retrosheet_df
        )

        combined_df = vector_constructor.match_game_moneylines_pipelined()

        return combined_df

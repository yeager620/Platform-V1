import unittest
from unittest.mock import patch
from datetime import datetime
import pandas as pd

# Assuming the DataPipeline class is defined in data_pipeline.py
from MaqAnalytics.VectorConstructor.DataPipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):

    def test_print_game_vectors_single_day(self):
        # Test Date
        test_date = "2023-06-20"

        # Initialize data pipeline instance
        pipeline = DataPipeline(start_date=test_date, end_date=test_date)

        # Generate game vectors for test date
        result_df = pipeline.process_games()
        self.assertIsNotNone(result_df)

        # Print game vector dataframe
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 3,
                               ):
            print(result_df)

        result_df.to_csv("/Users/yeager/Desktop/Maquoketa-Platform-V1/unit-tests/test_game_vector_df_2023-06-20.csv",
                         index=False, mode='w')

    # TODO: Implement actual unit test -- below is example implementaion
    def test_process_games_single_day(self):
        # Define the test date
        test_date = datetime.now().strftime("%m/%d/%Y")

        # Create example data frames to simulate the output of converters and scrapers
        example_retrosheet_data = pd.DataFrame({
            'Game_Date': [test_date],
            'Home_Team_Abbr': ['NYY'],
            'Away_Team_Abbr': ['BOS'],
            'Game_PK': [123456]
        })
        example_moneyline_data = pd.DataFrame({
            'date': [test_date],
            'team': ['NYY'],
            'opponent': ['BOS'],
            'odds': ['-150']
        })

        # Mock SavantRetrosheetConverter
        with patch('data_pipeline.SavantRetrosheetConverter') as MockConverter:
            mock_converter_instance = MockConverter.return_value
            mock_converter_instance.process_games_retrosheet_with_outcome.return_value = example_retrosheet_data

            # Mock SportsbookReviewScraper
            with patch('data_pipeline.SportsbookReviewScraper') as MockScraper:
                mock_scraper_instance = MockScraper.return_value
                mock_scraper_instance.scrape.return_value = example_moneyline_data

                # Mock VectorConstructor
                with patch('data_pipeline.VectorConstructor') as MockVectorConstructor:
                    mock_vector_instance = MockVectorConstructor.return_value
                    mock_vector_instance.match_game_moneylines_pipelined.return_value = pd.DataFrame({
                        'Game_Date': [test_date],
                        'Home_Team_Abbr': ['NYY'],
                        'Away_Team_Abbr': ['BOS'],
                        'Game_PK': [123456],
                        'odds': ['-150'],
                        'implied_odds': [0.60]
                    })

                    # Initialize DataPipeline
                    pipeline = DataPipeline(start_date=test_date, end_date=test_date)

                    # Execute process_games method
                    result_df = pipeline.process_games()

                    # Assertions to validate the expected results
                    self.assertIsNotNone(result_df)
                    self.assertEqual(result_df['Game_Date'].iloc[0], test_date)
                    self.assertEqual(result_df['Home_Team_Abbr'].iloc[0], 'NYY')
                    self.assertEqual(result_df['Away_Team_Abbr'].iloc[0], 'BOS')
                    print(result_df)


if __name__ == '__main__':
    unittest.main()

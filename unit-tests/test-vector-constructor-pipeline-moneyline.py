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

        print(f"Number of columns in combined_df: {result_df.shape[1]}")

        result_df.to_csv("/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/test_game_vector_df_2023-06-20.csv",
                         index=False, mode='w')

    def test_print_game_vectors_multiple_days(self):
        # Start Date
        start_date = "2023-07-02"
        end_date = "2023-08-01"

        # Initialize data pipeline instance
        pipeline = DataPipeline(start_date=start_date, end_date=end_date)

        # Generate game vectors for test date
        result_df = pipeline.process_games()
        self.assertIsNotNone(result_df)

        # Print game vector dataframe
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 3,
                               ):
            print(result_df)

        result_df.to_csv(f"/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.1-month_intervals/test_game_vector_df_{start_date}_{end_date}.csv",
                         index=False, mode='w')


if __name__ == '__main__':
    unittest.main()

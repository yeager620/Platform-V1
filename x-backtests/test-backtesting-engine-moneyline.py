# File: unit-tests/test-backtesting-engine-moneyline.py

import unittest
import pandas as pd
from MaqAnalytics.BacktestingEngine import BacktestingEngine  # Adjust the import path as necessary


class TestBacktestingEngineXGBoost(unittest.TestCase):
    def setUp(self):
        """
        Set up a mock dataset for testing.
        """
        # Read the cleaned mock data into a DataFrame
        self.mock_data = pd.read_csv(
            "/y-data/v1.1-full/game_vectors_01_2023-04-01_2024-11-15.csv")

        # Ensure 'park_id' is treated as categorical by converting it to string
        self.mock_data['park_id'] = self.mock_data['park_id'].astype(str)

        # Define target and moneyline columns
        self.target_column = 'Home_Win'
        self.moneyline_columns = ['home_odds', 'away_odds']

    def test_xgboost_model_backtest(self):
        """
        Test the BacktestingEngine with XGBoost model on mock data.
        """
        try:
            engine = BacktestingEngine(
                data=self.mock_data,
                target_column=self.target_column,
                moneyline_columns=self.moneyline_columns,
                model_type='xgboost',
                initial_train_size=0.5,
                random_state=28
            )

            # Run the full pipeline with an initial bankroll of $10,000
            results = engine.run_full_pipeline(initial_bankroll=10000.0)

            # Access backtest evaluation
            backtest_evaluation = results['Backtest_Evaluation']
            backtest_results = results['Backtest_Results']

            # Print the evaluation metrics
            print("Backtest Evaluation Metrics:")
            for key, value in backtest_evaluation.items():
                print(f"{key}: {value}")

            # Print classification metrics
            print("\nModel Accuracy Metrics:")
            for key, value in results['Accuracy_Metrics'].items():
                print(f"{key}: {value}")

            print("\nClassification Report:")
            print(results['Classification_Report'])

            print("\nConfusion Matrix:")
            print(results['Confusion_Matrix'])

            print("\nBacktest Results:")
            print(backtest_results)

            # Ensure that the final bankroll is greater than zero
            final_bankroll = backtest_evaluation.get('Final Bankroll', 0)
            self.assertGreater(final_bankroll, 0,
                               "Final bankroll should be greater than zero.")

            # Ensure that evaluation metrics are present
            self.assertIn('Total Profit', backtest_evaluation, "Backtest evaluation should contain 'Total Profit'.")
            self.assertIn('Return on Investment (ROI %)', backtest_evaluation,
                          "Backtest evaluation should contain 'Return on Investment (ROI %)'.")
            self.assertIn('Final Bankroll', backtest_evaluation, "Backtest evaluation should contain 'Final Bankroll'.")

            # Ensure that accuracy metrics are present
            self.assertIn('Accuracy', results['Accuracy_Metrics'], "Accuracy metrics should contain 'Accuracy'.")
            self.assertIn('ROC-AUC', results['Accuracy_Metrics'], "Accuracy metrics should contain 'ROC-AUC'.")

        except Exception as e:
            self.fail(f"BacktestingEngine with XGBoost raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()

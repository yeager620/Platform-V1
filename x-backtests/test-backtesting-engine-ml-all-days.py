import unittest
import pandas as pd

# IMPORTANT:
# Replace this import path with the actual module & class name of your "all games" engine
# For example, if you saved the new engine as "BacktestingEngineAllGames"
# in MaqAnalytics/MaqHistorical/BacktestingEngineAllGames.py, you might do:
# from MaqAnalytics.MaqHistorical.BacktestingEngineAllGames import BacktestingEngineAllGames

from MaqAnalytics.MaqHistorical.BacktestingEngineAllGames import BacktestingEngine


class TestBacktestingEngineXGBoostAllGames(unittest.TestCase):
    def setUp(self):
        """
        Set up a mock dataset for testing.
        """
        # Read the cleaned mock data into a DataFrame
        self.mock_data = pd.read_csv(
            "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.2-full/v1.2.7-game-vectors-ml-half_2021-04-01_2024-10-30.csv"
        )

        # Ensure 'park_id' is treated as categorical by converting it to string
        self.mock_data['park_id'] = self.mock_data['park_id'].astype(str)

        # Define target and moneyline columns
        self.target_column = 'Home_Win_Half'
        self.moneyline_columns = ['home_odds', 'away_odds']

    def test_xgboost_model_backtest_all_games(self):
        """
        Test the BacktestingEngineAllGames with XGBoost model on mock data.
        This version places bets on ALL games each day rather than just one.
        """
        try:
            engine = BacktestingEngine(
                data=self.mock_data,
                target_column=self.target_column,
                moneyline_columns=self.moneyline_columns,
                model_type='xgboost',
                initial_train_size=0.75,
                update_model=False,
                random_state=28,
                kelly_fraction=1
            )

            # Run the full pipeline with an initial bankroll of $10,000
            results = engine.run_full_pipeline(initial_bankroll=10000.0)

            # Access backtest evaluation
            backtest_evaluation = results['Backtest_Evaluation']
            backtest_results = results['Backtest_Results']

            # Print the evaluation metrics
            print("Backtest Evaluation Metrics (Betting All Games):")
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

            print("\nBacktest Results (first few rows):")
            print(backtest_results.head())

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
            self.fail(f"BacktestingEngineAllGames with XGBoost raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()

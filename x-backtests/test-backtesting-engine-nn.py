import unittest
import pandas as pd

from MaqHistorical.NN_BacktestingEngine import BacktestingEngine  # Example placeholder import


class TestBacktestingEngineNeuralNetwork(unittest.TestCase):
    def setUp(self):
        """
        Set up a mock dataset for testing the neural network backtester.
        """
        # Read or generate mock data into a DataFrame
        # Adjust the path to your local CSV or dataset
        self.mock_data = pd.read_csv(
            "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.2-full/v1.2.7-game-vectors-ml-half_2021-04-01_2024-10-30.csv"
        )

        # Convert 'park_id' to string for categorical processing if needed
        if 'park_id' in self.mock_data.columns:
            self.mock_data['park_id'] = self.mock_data['park_id'].astype(str)

        # Define target and moneyline columns
        self.target_column = 'Home_Win_Half'
        self.moneyline_columns = ['home_odds', 'away_odds']

    def test_neural_network_model_backtest(self):
        """
        Test the BacktestingEngine with a neural network model on mock data.
        """
        try:
            # Instantiate the NN-based backtesting engine
            engine = BacktestingEngine(
                data=self.mock_data,
                target_column=self.target_column,
                moneyline_columns=self.moneyline_columns,
                initial_train_size=0.75,   # 75% train, 25% backtest
                update_model=True,         # daily model updates
                random_state=28,
                kelly_fraction=1           # Full Kelly
            )

            # Run the full pipeline with an initial bankroll of $10,000
            results = engine.run_full_pipeline(initial_bankroll=10000.0)

            # Extract results
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
            print(backtest_results.head(10))  # Print first 10 rows for brevity

            # Basic assertions
            final_bankroll = backtest_evaluation.get('Final Bankroll', 0)
            self.assertGreater(final_bankroll, 0,
                               "Final bankroll should be greater than zero.")

            self.assertIn('Total Profit', backtest_evaluation,
                          "Backtest evaluation should contain 'Total Profit'.")
            self.assertIn('Return on Investment (ROI %)', backtest_evaluation,
                          "Backtest evaluation should contain 'Return on Investment (ROI %)'.")
            self.assertIn('Final Bankroll', backtest_evaluation,
                          "Backtest evaluation should contain 'Final Bankroll'.")

            self.assertIn('Accuracy', results['Accuracy_Metrics'],
                          "Accuracy metrics should contain 'Accuracy'.")
            self.assertIn('ROC-AUC', results['Accuracy_Metrics'],
                          "Accuracy metrics should contain 'ROC-AUC'.")

        except Exception as e:
            self.fail(f"BacktestingEngine with Neural Network raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()

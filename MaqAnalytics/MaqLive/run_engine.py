import os
from datetime import datetime
from Engine import Engine
import pandas as pd


def main():
    # Define paths and parameters
    dataset_path = "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.2-full/v1.2.7-game-vectors-ml-half_2021-04-01_2024-10-30.csv"
    target_column = "Home_Win_Half"
    moneyline_columns = ["home_odds", "away_odds"]  # Replace with your actual moneyline column names

    # to fix or mock the current time for testing:
    # current_time = datetime(2023, 5, 1, 12, 0, 0)

    engine = Engine(
        dataset_path=dataset_path,
        target_column=target_column,
        moneyline_columns=moneyline_columns,
        days_ahead=7,
        max_concurrent_requests=10,
        update_interval=3600,  # 1 hour in seconds
        model_type="xgboost",
        output_folder="/Users/yeager/Desktop/Maquoketa-Platform-V1/MaqAnalytics/MaqLive/live-reports",
        live_data_folder="/Users/yeager/Desktop/Maquoketa-Platform-V1/MaqAnalytics/MaqLive/live-data",
        random_state=28,
        # current_time=current_time,  # Uncomment to force a specific datetime
    )

    # Run the Engine
    engine.run()  # This starts a background thread to update & predict every hour


if __name__ == "__main__":
    main()
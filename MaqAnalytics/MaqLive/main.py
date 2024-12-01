# main.py

import os
from .Engine import Engine  # Assuming the Engine class is in engine.py
import pandas as pd


def main():
    # Define paths and parameters
    dataset_path = "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.1-full/v1.1-n4775-game-vectors_2021-04-01_2024-10-30.csv"  # Path to your existing dataset
    target_column = "Home_Win"  # Replace with your actual target column name
    moneyline_columns = ["home_odds", "away_odds"]  # Replace with your actual moneyline column names

    # Initialize the Engine
    engine = Engine(
        dataset_path=dataset_path,
        target_column=target_column,
        moneyline_columns=moneyline_columns,
        days_ahead=100,
        max_concurrent_requests=10,
        update_interval=300,  # 5 minutes
        model_type="xgboost",
        initial_train_size=0.2,
        kelly_fraction=1.0,
        output_folder="./reports",
        random_state=42
    )

    # Run the Engine
    engine.run()


if __name__ == "__main__":
    main()

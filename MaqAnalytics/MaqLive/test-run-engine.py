import os
from datetime import datetime
from Engine import Engine


def run_engine():
    """
    Test run of the Engine class with the date fixed to 2024-10-30.
    This is useful for confirming model predictions for a past or future date.
    """
    # Define paths and parameters
    dataset_path = "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.2-full/v1.2.6-game-vectors_2021-04-01_2024-10-30.csv"
    target_column = "Home_Win"
    moneyline_columns = ["home_odds", "away_odds"]

    # Here we fix "today" to 2024-10-30 via current_time.
    engine = Engine(
        dataset_path=dataset_path,
        target_column=target_column,
        moneyline_columns=moneyline_columns,
        days_ahead=7,
        max_concurrent_requests=10,
        update_interval=3600,  # 1 hour in seconds, but won't matter if we only run update_and_predict() once
        model_type="xgboost",
        output_folder="/Users/yeager/Desktop/Maquoketa-Platform-V1/MaqAnalytics/MaqLive/live-reports",
        live_data_folder="/Users/yeager/Desktop/Maquoketa-Platform-V1/MaqAnalytics/MaqLive/live-data",
        random_state=28,
        current_time=datetime(2024, 10, 30)  # <-- The key parameter for test/fixed date
    )

    # Perform a single update-and-predict cycle
    engine.update_and_predict()

    # Retrieve and print the predictions
    predictions = engine.get_predictions()
    print("Predictions for 2024-10-30:")
    for game_pk, game_data in predictions.items():
        print(f"Game_PK: {game_pk} | Data: {game_data}")


if __name__ == "__main__":
    run_engine()

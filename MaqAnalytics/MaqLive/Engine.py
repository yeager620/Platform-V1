# main.py

import os
import json
import datetime
import signal
import threading
import time
from typing import List, Dict, Any, Optional, Callable

import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

from VectorConstructor.DataPipeline import DataPipeline
from .LiveGamelogsFeed import LiveGamelogsFeed


class Engine:
    def __init__(
            self,
            dataset_path: str,
            target_column: str,
            moneyline_columns: list,
            days_ahead: int = 30,
            max_concurrent_requests: int = 10,
            update_interval: int = 300,  # seconds
            callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
            model_type: str = "xgboost",
            output_folder: str = "/Users/yeager/Desktop/Maquoketa-Platform-V1/MaqAnalytics/MaqLive/live-reports",
            random_state: int = 28,
    ):
        """
        Initialize the Engine class with model training and live feed setup.

        :param dataset_path: Path to the existing dataset CSV file.
        :param target_column: Name of the target column in the dataset.
        :param moneyline_columns: List containing names of moneyline columns, e.g., ['home_odds', 'away_odds'].
        :param days_ahead: Number of days ahead to look for upcoming games.
        :param max_concurrent_requests: Maximum number of concurrent HTTP requests.
        :param update_interval: Time interval between updates in seconds.
        :param callback: Optional function to call with updated gamelogs.
        :param model_type: Type of model to train ('xgboost', 'random_forest', etc.).
        :param output_folder: Directory to save reports and outputs.
        :param random_state: Random state for reproducibility.
        """
        # Load and prepare the dataset
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.moneyline_columns = moneyline_columns

        self.data = self.load_dataset()

        # Train the model on 100% of the data
        self.train_model()

        # Initialize LiveGamelogsFeed
        self.live_feed = LiveGamelogsFeed(
            days_ahead=days_ahead,
            max_concurrent_requests=max_concurrent_requests,
            update_interval=update_interval,
            callback=self.on_gamelogs_update  # Use internal callback
        )

        # Store parameters for reporting or further use
        self.model_type = model_type
        self.output_folder = output_folder
        self.random_state = random_state

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the existing dataset from a CSV file.

        :return: pandas DataFrame containing the dataset.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at path: {self.dataset_path}")

        data = pd.read_csv(self.dataset_path)
        print(f"Dataset loaded with {len(data)} records.")

        return data

    def initialize_model(self, model_type: str, random_state: int):
        """
        Initialize the machine learning model based on the specified type.

        :param model_type: Type of model to train ('xgboost', 'random_forest', etc.).
        :param random_state: Random state for reproducibility.
        :return: Initialized machine learning model.
        """
        model_type = model_type.lower()
        if model_type == "logistic_regression":
            model = LogisticRegression(random_state=random_state)
        elif model_type == "xgboost":
            model = xgb.XGBClassifier(
                eval_metric='logloss',
                random_state=random_state
            )
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")

        return model

    def train_model(self):
        """
        Prepare data and train the machine learning model.
        """
        print("Preparing data for training...")
        combined_df = self.live_game_predictor.prepare_data()

        print("Training the model on the entire dataset...")
        self.live_game_predictor.train_model(combined_df)
        print("Model training and calibration completed.")

        # Optionally, perform cross-validation
        self.perform_cross_validation(combined_df)

    def perform_cross_validation(self, combined_df: pd.DataFrame, cv_folds: int = 5):
        """
        Perform cross-validation to assess model performance.

        :param combined_df: DataFrame containing features and target variables.
        :param cv_folds: Number of cross-validation folds.
        """
        features = combined_df.drop(columns=["Home_Win", "Game_Date", "Game_PK", "Home_Team_Name", "Away_Team_Name"])
        target = combined_df["Home_Win"]

        # Define a pipeline consistent with LiveGamePredictor's preprocessing
        pipeline = self.live_game_predictor.pipeline

        print(f"Performing {cv_folds}-fold cross-validation...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(pipeline, features, target, cv=cv, scoring='roc_auc')

        print(f"Cross-validation ROC-AUC scores: {scores}")
        print(f"Mean ROC-AUC: {scores.mean():.4f} | Std: {scores.std():.4f}")

    def on_gamelogs_update(self, updated_gamelogs: List[Dict[str, Any]]):
        """
        Callback function that gets called when gamelogs are updated.

        :param updated_gamelogs: List of updated gamelog data.
        """
        print(f"Engine: Received {len(updated_gamelogs)} updated gamelogs.")
        self.process_all_upcoming(updated_gamelogs)

    def process_all_upcoming(self, gamelogs: List[Dict[str, Any]]):
        """
        Make predictions on the upcoming games and process the results.

        :param gamelogs: List of gamelog data for upcoming games.
        """
        predictions = {}
        for gamelog in gamelogs:
            game_pk = gamelog.get('scoreboard', {}).get('gamePk', None)
            if not game_pk:
                continue

            # Extract features required for prediction
            game_features = self.generate_feature_vector(gamelog)

            if game_features is None:
                print(f"Engine: Missing features for game PK {game_pk}. Skipping prediction.")
                continue

            # Predict probability of home win
            proba_home_win = self.predict_live_game(game_pk)

            # Make a prediction based on the probability
            predicted_outcome = 'home_win' if proba_home_win >= 0.5 else 'away_win'

            # Prepare prediction result
            predictions[game_pk] = {
                'prob_home_win': proba_home_win,
                'predicted_outcome': predicted_outcome,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def generate_feature_vector(self, gamelog):

        return

    def predict_live_game(self, game_pk):
        pass

    def run(self):
        """
        Run the Engine by starting the live gamelogs feed.
        """
        print("Starting Engine...")
        # Start the live feed in a separate thread to allow concurrent execution
        feed_thread = threading.Thread(target=self.live_feed.start_feed, daemon=True)
        feed_thread.start()

        print("Engine is running. Press Ctrl+C to stop.")
        try:
            while True:
                # Keep the main thread alive to allow background thread to run
                signal.pause()
        except AttributeError:
            # signal.pause() is not available on some platforms like Windows
            while True:
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    print("\nEngine stopped by user.")
                    break
        except KeyboardInterrupt:
            print("\nEngine stopped by user.")

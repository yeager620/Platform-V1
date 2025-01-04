import os
import datetime
from datetime import datetime, timedelta
import signal
import threading
import time
from typing import List, Dict, Any, Optional, Callable

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from transformers import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from MaqAnalytics.VectorConstructor.DataPipeline import DataPipeline


class Engine:
    def __init__(
            self,
            dataset_path: str,
            target_column: str,
            moneyline_columns: list,
            days_ahead: int = 7,
            max_concurrent_requests: int = 10,
            update_interval: int = 3600,  # seconds
            callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
            model_type: str = "xgboost",
            output_folder: str = "/Users/yeager/Desktop/Maquoketa-Platform-V1/MaqAnalytics/MaqLive/live-reports",
            live_data_folder: str = "/Users/yeager/Desktop/Maquoketa-Platform-V1/MaqAnalytics/MaqLive/live-data",
            random_state: int = 28,
            current_time: Optional[datetime] = None,
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
        :param live_data_folder: Directory to save live data.
        :param random_state: Random state for reproducibility.
        :param current_time: Optional datetime object to override "now" or "today".
                             If not provided, use the real current time.
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.moneyline_columns = moneyline_columns
        self.days_ahead = days_ahead
        self.update_interval = update_interval
        self.output_folder = output_folder
        self.live_data_folder = live_data_folder
        self.random_state = random_state
        self.model_type = model_type

        # If current_time is None, fall back to the actual current date/time
        self._current_time = current_time if current_time is not None else datetime.now()

        # Load and prepare the dataset
        self.data = self.load_dataset()

        # We'll use self.get_current_time() anytime we need "today" or "now"
        self.date = self.get_current_time()

        excluded_columns = self.moneyline_columns + [self.target_column, 'Game_Date', 'Game_PK']
        self.feature_columns = [col for col in self.data.columns if col not in excluded_columns]

        # Create a DataPipeline for the time window [today, today + days_ahead]
        start_date_str = self.get_current_time().strftime("%Y-%m-%d")
        end_date_str = (self.get_current_time() + timedelta(days=self.days_ahead)).strftime("%Y-%m-%d")
        self.data_pipeline = DataPipeline(start_date_str, end_date_str)
        if not self.data_pipeline.savant_converter.gamelogs:
            print(f"MaqLive Engine: No games found for the next {self.days_ahead} days")

        try:
            # Attempt to create feature vectors for upcoming games (unlabeled data)
            self.unlabeled_vector_df = self.data_pipeline.process_games()

            # Train the model on historical data
            self.preprocess_data()
            self.initialize_model(model_type, random_state)
            self.train_model()

            # Predictions DataFrame will be stored here
            self.predictions_df = pd.DataFrame()

        except Exception as e:
            print(f"{len(self.data_pipeline.savant_converter.gamelogs)} upcoming games found")
            print(f"Error that occurred: {e}")

    def get_current_time(self) -> datetime:
        """
        Returns the current engine time.
        If 'current_time' was given, it returns that fixed time; otherwise returns now().
        """
        return self._current_time

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the existing dataset from a CSV file, but only keep rows where Game_Date < current engine time.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at path: {self.dataset_path}")

        data = pd.read_csv(self.dataset_path)
        print(f"Dataset loaded with {len(data)} records before date filtering.")

        # Convert Game_Date to datetime
        if 'Game_Date' not in data.columns:
            raise ValueError("Data must contain a 'Game_Date' column.")

        data['Game_Date'] = pd.to_datetime(data['Game_Date'], errors='coerce')
        data = data.dropna(subset=['Game_Date'])

        # Keep only rows with Game_Date strictly before current_time
        current_time = self.get_current_time()
        original_len = len(data)
        data = data[data['Game_Date'] < current_time]
        filtered_len = len(data)

        print(f"Filtered out {original_len - filtered_len} future games. "
              f"{filtered_len} remain for training.")
        return data

    def preprocess_data(self):
        """
        Preprocess self.data by sorting, dropping missing moneyline rows, and building transforms.
        """
        # Sort by date (already chronological, but just to be safe)
        self.data.sort_values(by='Game_Date', inplace=True)

        # Convert park_id to string if present
        if 'park_id' in self.data.columns:
            self.data['park_id'] = self.data['park_id'].astype(str)

        # Drop rows with missing moneyline values
        self.data.dropna(subset=self.moneyline_columns, inplace=True)

        # Separate features and target
        X = self.data.drop(columns=[self.target_column, "Game_Date", "Game_PK"], errors='ignore')
        y = self.data[self.target_column]

        # Identify numerical and categorical columns
        numerical_cols = [
            col for col in X.select_dtypes(include=["int64", "float64"]).columns
            if col not in self.moneyline_columns
        ]
        categorical_cols = [
            col for col in X.select_dtypes(include=["object", "category", "bool"]).columns
            if col not in self.moneyline_columns
        ]

        # Build the transformers
        numerical_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Combine them in a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        self.preprocessor = preprocessor
        self.X = X
        self.y = y

    def initialize_model(self, model_type: str, random_state: int):
        """
        Initialize the pipeline with a CalibratedClassifierCV around the chosen base model.
        """
        model_type = model_type.lower()
        if model_type == "xgboost":
            # Create base XGB model
            base_xgb = xgb.XGBClassifier(
                eval_metric='logloss',
                random_state=random_state
            )
            # Wrap it with CalibratedClassifierCV
            calibrated_model = CalibratedClassifierCV(
                base_estimator=base_xgb,
                method='sigmoid',
                cv=3
            )
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")

        # Build an overall pipeline with preprocessing + calibrator
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", calibrated_model)
        ])

    def train_model(self):
        """
        Hyperparameter-tune the pipeline.
        For XGBoost inside a CalibratedClassifierCV, reference
        classifier__base_estimator__<param> for XGB's hyperparams.
        """
        if self.model_type == "xgboost":
            # Note that we reference the XGB parameters through base_estimator
            param_grid = {
                "classifier__base_estimator__n_estimators": [50, 75, 100],
                "classifier__base_estimator__max_depth": [2, 3],
                "classifier__base_estimator__learning_rate": [0.05, 0.1],
            }
        else:
            return  # Insert other models' grids if needed

        print("Starting hyperparameter tuning on initial training data...")
        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=6,
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X, self.y)
        self.best_params = grid_search.best_params_
        print("Best hyperparameters found:", grid_search.best_params_)

        # Save the best estimator pipeline
        self.pipeline = grid_search.best_estimator_

        # The pipeline now includes the best hyperparameters
        # in the XGB base_estimator (wrapped by CalibratedClassifierCV).
        print("Hyperparameter tuning and calibration complete.\n")

    def perform_cross_validation(self, combined_df: pd.DataFrame, cv_folds: int = 5):
        """
        Optionally, you can run cross-validation on combined historical data
        to assess performance.
        """
        features = combined_df.drop(columns=["Home_Win", "Game_Date", "Game_PK"], errors='ignore')
        target = combined_df["Home_Win"]

        pipeline = self.pipeline
        print(f"Performing {cv_folds}-fold cross-validation...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(pipeline, features, target, cv=cv, scoring='roc_auc')

        print(f"Cross-validation ROC-AUC scores: {scores}")
        print(f"Mean ROC-AUC: {scores.mean():.4f} | Std: {scores.std():.4f}")

    def generate_feature_vectors(self):
        """
        Pull new gamelogs, identify newly-finished games,
        append them to the dataset, and re-train if needed.
        """
        self.unlabeled_vector_df = self.data_pipeline.process_games()
        new_finished_games = self.unlabeled_vector_df[
            self.unlabeled_vector_df[self.target_column].isin([0, 1])
        ]
        if not new_finished_games.empty:
            print(f"Found {len(new_finished_games)} new finished games.")

            # Remove those completed games from unlabeled
            self.unlabeled_vector_df = self.unlabeled_vector_df[
                ~self.unlabeled_vector_df['Game_PK'].isin(new_finished_games['Game_PK'])
            ]

            # Ensure correct datetime type
            new_finished_games['Game_Date'] = pd.to_datetime(new_finished_games['Game_Date'], errors='coerce')

            # Load the existing dataset again
            existing_data = self.load_dataset()
            existing_data['Game_Date'] = pd.to_datetime(existing_data['Game_Date'], errors='coerce')

            # Concatenate old + new
            combined_data = pd.concat([existing_data, new_finished_games], ignore_index=True)
            combined_data['Game_Date'] = pd.to_datetime(combined_data['Game_Date'], errors='coerce')
            combined_data.sort_values(by='Game_Date', inplace=True)

            # Save updated file
            combined_data.to_csv(self.dataset_path, index=False)
            print("New finished games appended to the dataset.")

            # Re-load and re-train
            self.data = self.load_dataset()
            self.data['Game_Date'] = pd.to_datetime(self.data['Game_Date'], errors='coerce')
            self.preprocess_data()
            self.train_model()
            print("Model retrained with new data.")

        # Sort unlabeled upcoming data by date for consistency
        if self.unlabeled_vector_df is not None:
            self.unlabeled_vector_df['Game_Date'] = pd.to_datetime(
                self.unlabeled_vector_df['Game_Date'], errors='coerce'
            )
            self.unlabeled_vector_df.sort_values(by='Game_Date', inplace=True)

    def predict_live_games(self):
        """
        Predict the probabilities of Home_Win on upcoming games.
        """
        if self.unlabeled_vector_df is None or self.pipeline is None:
            print("No data to predict or model not trained.")
            return

        # We remove target columns before prediction (since these are unlabeled in practice)
        X_live = self.unlabeled_vector_df.drop(
            columns=[self.target_column, "Game_Date", "Game_PK"], errors='ignore'
        )

        # Predict home-win probabilities
        probabilities = self.pipeline.predict_proba(X_live)[:, 1]
        self.unlabeled_vector_df['Home_Win'] = probabilities
        self.unlabeled_vector_df.sort_values(by='Game_Date', inplace=True)

        # Build a predictions_df with relevant columns
        self.predictions_df = self.unlabeled_vector_df[[
            'Game_PK', 'Game_Date', 'Home_Team_Abbr', 'Away_Team_Abbr',
            self.moneyline_columns[0], self.moneyline_columns[1],
            'home_odds_decimal', 'away_odds_decimal',
            'home_implied_prob', 'away_implied_prob',
            'home_wager_percentage', 'away_wager_percentage',
            'Home_Win'
        ]].copy()

        # Rename the 'Home_Win' column to clarify it's a probability
        self.predictions_df.rename(columns={'Home_Win': 'Home_Win_Prob_Theo'}, inplace=True)
        self.predictions_df['Away_Win_Prob_Theo'] = 1 - self.predictions_df['Home_Win_Prob_Theo']

        # Save predictions to disk
        output_file = os.path.join(self.live_data_folder, "live_predictions.csv")
        os.makedirs(self.output_folder, exist_ok=True)
        self.predictions_df.to_csv(output_file, index=False)
        print(f"Updated {output_file}")

    def update_and_predict(self):
        """
        Refresh data (both finished & upcoming) and produce new predictions.
        """
        print("Updating data and generating new predictions...")

        # You might optionally refresh self._current_time here if desired
        # For a true live system, you'd do self._current_time = datetime.now()

        start_date_str = self.get_current_time().strftime("%Y-%m-%d")
        end_date_str = (self.get_current_time() + timedelta(days=self.days_ahead)).strftime("%Y-%m-%d")

        # Recreate pipeline for upcoming data
        self.data_pipeline = DataPipeline(start_date_str, end_date_str, version=2)

        # Possibly see if new games are finished, update dataset, re-train
        self.generate_feature_vectors()

        # Then predict outcomes for upcoming games
        self.predict_live_games()

    def get_predictions(self) -> Dict[int, Dict]:
        """
        Return predictions as a dictionary: { Game_PK: {...fields...}, ... }
        """
        if self.predictions_df is not None and not self.predictions_df.empty:
            predictions_dict = self.predictions_df.set_index('Game_PK').to_dict('index')
            return predictions_dict
        else:
            print("No predictions available.")
            return {}

    def run(self):
        """
        Run the Engine by updating & predicting in a loop (daemon thread).
        """
        print("Starting Engine...")

        def periodic_update():
            while True:
                self.update_and_predict()
                time.sleep(self.update_interval)

        update_thread = threading.Thread(target=periodic_update, daemon=True)
        update_thread.start()

        print(f"Engine is running. Updates every {self.update_interval} seconds. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nEngine stopped by user.")

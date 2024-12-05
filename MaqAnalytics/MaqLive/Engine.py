import os
import datetime
from datetime import datetime
import signal
import threading
import time
from typing import List, Dict, Any, Optional, Callable

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
        self.days_ahead = days_ahead
        self.update_interval = update_interval

        self.data = self.load_dataset()

        excluded_columns = self.moneyline_columns + [self.target_column, 'Game_Date', 'Game_PK']
        self.feature_columns = [col for col in self.data.columns if col not in excluded_columns]

        self.data_pipeline = DataPipeline(datetime.date.today().strftime("%Y-%m-%d"), (datetime.date.today() + datetime.timedelta(days=7)).strftime("%Y-%m-%d"))
        self.unlabeled_vector_df = self.data_pipeline.process_games()

        # Train the model on 100% of the data
        self.preprocess_data()
        self.initialize_model(model_type, random_state)
        self.train_model()

        # Store parameters for reporting or further use
        self.model_type = model_type
        self.output_folder = output_folder
        self.random_state = random_state

        self.pipeline = None
        self.preprocessor = None

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

    def preprocess_data(self):
        # Preprocessing code remains the same
        if 'Game_Date' in self.data.columns:
            self.data.sort_values(by='Game_Date', inplace=True)
        else:
            raise ValueError("Data must contain a 'Game_Date' column for chronological sorting.")

        # Ensure 'park_id' is treated as a categorical variable by converting it to string
        if 'park_id' in self.data.columns:
            self.data['park_id'] = self.data['park_id'].astype(str)

        # Drop rows with missing moneyline values
        self.data.dropna(subset=self.moneyline_columns, inplace=True)

        # Separate features and target
        X = self.data.drop(columns=[self.target_column, "Game_Date", "Game_PK"])
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

        # Define preprocessing steps
        numerical_transformer = Pipeline(steps=[
            # ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # Save the preprocessor for future use
        self.preprocessor = preprocessor

        # Save features and target
        self.X = X
        self.y = y

        # Save feature names after preprocessing
        # self.feature_names = self.get_feature_names()

    def initialize_model(self, model_type: str, random_state: int):
        """
        Initialize the machine learning model based on the specified type.

        :param model_type: Type of model to train ('xgboost', 'random_forest', etc.).
        :param random_state: Random state for reproducibility.
        :return: Initialized machine learning model.
        """
        model_type = model_type.lower()
        if model_type == "xgboost":
            base_model = xgb.XGBClassifier(
                eval_metric='logloss',
                random_state=random_state
            )
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")

        calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)

        # Create a pipeline that first preprocesses the data and then fits the calibrated model
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("calibrated_classifier", calibrated_model)
        ])

    def train_model(self):
        """
        Train the model.
        """
        self.pipeline.fit(self.X, self.y)
        print("Model training completed.")

    def perform_cross_validation(self, combined_df: pd.DataFrame, cv_folds: int = 5):
        """
        Perform cross-validation to assess model performance.

        :param combined_df: DataFrame containing features and target variables.
        :param cv_folds: Number of cross-validation folds.
        """
        features = combined_df.drop(columns=["Home_Win", "Game_Date", "Game_PK"])
        target = combined_df["Home_Win"]

        # Define a pipeline consistent with LiveGamePredictor's preprocessing
        pipeline = self.pipeline

        print(f"Performing {cv_folds}-fold cross-validation...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(pipeline, features, target, cv=cv, scoring='roc_auc')

        print(f"Cross-validation ROC-AUC scores: {scores}")
        print(f"Mean ROC-AUC: {scores.mean():.4f} | Std: {scores.std():.4f}")

    def generate_feature_vectors(self):
        self.unlabeled_vector_df = self.data_pipeline.process_games()
        new_finished_games = self.unlabeled_vector_df[self.unlabeled_vector_df['home_win'].isin([0, 1])]
        if not new_finished_games.empty:
            print(f"Found {len(new_finished_games)} new finished games.")

            # Remove finished games from unlabeled_vector_df
            self.unlabeled_vector_df = self.unlabeled_vector_df[
                ~self.unlabeled_vector_df['Game_PK'].isin(new_finished_games['Game_PK'])
            ]

            # Append new finished games to the dataset CSV file
            # Load existing dataset
            existing_data = self.load_dataset()

            # Concatenate existing data with new finished games
            combined_data = pd.concat([existing_data, new_finished_games], ignore_index=True)

            # Sort chronologically
            combined_data.sort_values(by='Game_Date', inplace=True)

            # Save back to CSV
            combined_data.to_csv(self.dataset_path, index=False)
            print("New finished games appended to the dataset.")

            # Re-load and preprocess data
            self.data = self.load_dataset()
            self.preprocess_data()

            # Re-train the model
            self.train_model()
            print("Model retrained with new data.")

            # Sort unlabeled_vector_df for upcoming predictions
        if self.unlabeled_vector_df is not None:
            self.unlabeled_vector_df.sort_values(by='Game_Date', inplace=True)

    def predict_live_games(self):
        """
        Predict the outcomes of upcoming games.
        """
        if self.unlabeled_vector_df is None or self.pipeline is None:
            print("No data to predict or model not trained.")
            return

        # Prepare the data (unlabeled data)
        X_live = self.unlabeled_vector_df.drop(columns=[self.target_column, "Game_Date", "Game_PK"],
                                               errors='ignore')

        # Predict probabilities using the pipeline (which includes preprocessing)
        probabilities = self.pipeline.predict_proba(X_live)[:, 1]  # Probability of Home_Win == 1

        # Add the probabilities to the DataFrame
        self.unlabeled_vector_df['Home_Win'] = probabilities
        self.unlabeled_vector_df.sort_values(by='Game_Date', inplace=True)

    def update_and_predict(self):
        """
        Method to update data and make predictions. This replaces the previous callback mechanism.
        """
        print("Updating data and generating new predictions...")
        # Update the data pipeline's date range
        self.data_pipeline = DataPipeline(
            datetime.date.today().strftime("%Y-%m-%d"),
            (datetime.date.today() + datetime.timedelta(days=self.days_ahead)).strftime("%Y-%m-%d")
        )
        self.generate_feature_vectors()
        self.predict_live_games()

    def run(self):
        """
        Run the Engine by periodically updating data and making predictions.
        """
        print("Starting Engine...")

        def periodic_update():
            while True:
                self.update_and_predict()
                time.sleep(self.update_interval)

        # Start the periodic update in a separate thread
        update_thread = threading.Thread(target=periodic_update, daemon=True)
        update_thread.start()

        print(f"Engine is running. Updates every {self.update_interval} seconds. Press Ctrl+C to stop.")
        try:
            while True:
                # Keep the main thread alive
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nEngine stopped by user.")

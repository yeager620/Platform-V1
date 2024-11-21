from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pandas as pd

from VectorConstructor.DataPipeline import DataPipeline


class LiveGamePredictor:
    def __init__(self, data_pipeline: DataPipeline, model):
        """
        Initializes the LiveGamePredictor.

        Parameters:
            data_pipeline (DataPipeline): An instance of the DataPipeline class for data processing.
        """
        self.data_pipeline = data_pipeline
        self.model = model
        self.calibrated_model = None

    def prepare_data(self) -> pd.DataFrame:
        """
        Prepares the data by processing games and returning a DataFrame with features and targets.

        Returns:
            pd.DataFrame: The combined data with features and target variables.
        """
        combined_df = self.data_pipeline.process_games()
        return combined_df

    def train_model(self, combined_df: pd.DataFrame):
        """
        Trains and calibrates the model for live predictions.

        Parameters:
            combined_df (pd.DataFrame): DataFrame containing features and target variables.
        """
        features = combined_df.drop(columns=["Home_Win", "Game_Date", "Game_PK", "Home_Team_Name", "Away_Team_Name"])
        target = combined_df["Home_Win"]

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        # Calibrate the model for better probability prediction
        self.calibrated_model = CalibratedClassifierCV(self.model, method='sigmoid')
        self.calibrated_model.fit(X_train, y_train)

        print(f"Model training complete. Accuracy on test set: {self.calibrated_model.score(X_test, y_test):.2f}")

    def construct_feature_vector(self, game_pk: int) -> pd.DataFrame:
        """
        Constructs a feature vector for the given gamePk.

        Parameters:
            game_pk (int): The unique identifier for the game.

        Returns:
            pd.DataFrame: The feature vector for the game.
        """
        # Fetch game data using SavantRetrosheetConverter
        game_data = self.data_pipeline.savant_converter.fetch_gamelog(game_pk)
        if not game_data:
            raise ValueError(f"No data found for gamePk {game_pk}.")

        # Process the game data into a feature vector
        processed_data = self.data_pipeline.savant_converter.process_games_retrosheet()
        feature_vector = processed_data[processed_data["Game_PK"] == game_pk]

        if feature_vector.empty:
            raise ValueError(f"Feature vector construction failed for gamePk {game_pk}.")

        # Drop columns not used for prediction
        feature_vector = feature_vector.drop(columns=["Home_Win", "Game_Date", "Game_PK", "Home_Team_Name", "Away_Team_Name"])

        return feature_vector

    def predict_live_game(self, game_pk: int) -> float:
        """
        Predicts the probability of the home team winning for the given gamePk.

        Parameters:
            game_pk (int): The unique identifier for the game.

        Returns:
            float: The predicted probability of a home win.
        """
        if not self.calibrated_model:
            raise ValueError("The model has not been trained or calibrated yet.")

        feature_vector = self.construct_feature_vector(game_pk)
        prediction_proba = self.calibrated_model.predict_proba(feature_vector)

        return prediction_proba[0, 1]  # Probability of Home_Win = 1

    @staticmethod
    def calculate_kelly_fraction(win_probability: float, odds: float) -> float:
        """
        Calculates the optimal Kelly fraction for betting.

        Parameters:
            win_probability (float): The probability of the predicted event occurring.
            odds (float): The decimal odds offered by the bookmaker (e.g., 2.5 for +150 moneyline).

        Returns:
            float: The optimal fraction of the bankroll to wager using the Kelly criterion.
        """
        # Convert decimal odds to net odds (return per unit wagered)
        net_odds = float(odds) - 1  # Ensure net_odds is a float

        # Kelly criterion formula: f* = (bp - q) / b
        # where:
        #   b = net odds,
        #   p = probability of winning,
        #   q = probability of losing = 1 - p
        losing_probability = 1 - float(win_probability)
        kelly_fraction = (net_odds * win_probability - losing_probability) / net_odds

        # Ensure the Kelly fraction is non-negative
        return max(0.0, kelly_fraction)


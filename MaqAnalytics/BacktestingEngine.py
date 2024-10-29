import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import warnings
import math

warnings.filterwarnings("ignore")  # To suppress warnings for cleaner output


class BacktestingEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        moneyline_columns: list,
        model_type: str = "logistic_regression",
        initial_train_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initializes the BacktestingEngine with the dataset and model parameters.

        Parameters:
            data (pd.DataFrame): The dataset containing feature vectors and target variable.
            target_column (str): The name of the target variable column.
            moneyline_columns (list): List containing the names of the moneyline columns [home_moneyline, away_moneyline].
            model_type (str): The type of model to use. Defaults to 'logistic_regression'.
            initial_train_size (float): Proportion of the dataset to use for initial training before backtesting. Defaults to 0.1 (10%).
            random_state (int): Controls the shuffling applied to the data before splitting. Defaults to 42.
        """
        self.data = data.copy()
        self.target_column = target_column
        self.moneyline_columns = moneyline_columns  # [home_moneyline, away_moneyline]
        self.model_type = model_type.lower()
        self.initial_train_size = initial_train_size
        self.random_state = random_state

        # Initialize placeholders
        self.model = None
        self.pipeline = None
        self.initial_train_data = None
        self.backtest_data = None

        # Preprocessing and model selection
        self.preprocess_data()
        self.split_data()
        self.select_model()

    def preprocess_data(self):
        """
        Preprocesses the data by handling missing values, encoding categorical variables,
        and scaling numerical features.
        """
        # Sort data chronologically based on 'date' column
        if 'date' in self.data.columns:
            self.data.sort_values(by='date', inplace=True)
        elif 'game_date' in self.data.columns:
            self.data.sort_values(by='game_date', inplace=True)
        else:
            raise ValueError("Data must contain a 'date' or 'game_date' column for chronological sorting.")

        # Separate features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        # Exclude moneyline columns from categorical encoding
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in self.moneyline_columns]

        # Define preprocessing steps
        numerical_transformer = Pipeline(steps=[
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

    def split_data(self):
        """
        Splits the data into initial training and backtesting sets based on chronological order.
        """
        total_games = len(self.data)
        initial_train_end = int(total_games * self.initial_train_size)

        self.initial_train_data = self.data.iloc[:initial_train_end].copy()
        self.backtest_data = self.data.iloc[initial_train_end:].copy()

        # Separate features and target for initial training
        X_train = self.initial_train_data.drop(columns=[self.target_column])
        y_train = self.initial_train_data[self.target_column]

        self.X_train = X_train
        self.y_train = y_train

    def select_model(self):
        """
        Selects and initializes the machine learning model based on the specified model_type.
        Defaults to Logistic Regression. Integrates probability calibration.
        """
        if self.model_type == "logistic_regression":
            base_model = LogisticRegression(random_state=self.random_state)
        elif self.model_type == "random_forest":
            base_model = RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == "gradient_boosting":
            base_model = GradientBoostingClassifier(random_state=self.random_state)
        elif self.model_type == "svm":
            base_model = SVC(probability=True, random_state=self.random_state)
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported.")

        # Wrap the base model with CalibratedClassifierCV for probability calibration
        calibrated_model = CalibratedClassifierCV(base_estimator=base_model, method='sigmoid', cv=5)

        # Create a pipeline that first preprocesses the data and then fits the calibrated model
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("calibrated_classifier", calibrated_model)
        ])

    def train_model(self):
        """
        Trains the machine learning model using the initial training data.
        """
        self.pipeline.fit(self.X_train, self.y_train)

    def update_model(self, X_new, y_new):
        """
        Updates the model with new data by retraining. Since CalibratedClassifierCV does not support partial_fit,
        the entire model is retrained with the existing and new data.

        Parameters:
            X_new (pd.DataFrame): New feature data.
            y_new (pd.Series): New target data.
        """
        # Append new data to training set
        self.X_train = pd.concat([self.X_train, X_new], ignore_index=True)
        self.y_train = pd.concat([self.y_train, y_new], ignore_index=True)

        # Retrain the pipeline (includes recalibrating probabilities)
        self.pipeline.fit(self.X_train, self.y_train)

    def predict_proba_single_game(self, game_features):
        """
        Predicts the probability of the home team winning for a single game.

        Parameters:
            game_features (pd.DataFrame): Feature data for the game.

        Returns:
            float: Probability of the home team winning.
        """
        proba = self.pipeline.predict_proba(game_features)[0][1]  # Assuming positive class is home win
        return proba

    def run_backtest(self, initial_bankroll: float = 10000.0):
        """
        Executes the backtesting simulation using the Kelly betting strategy.

        Parameters:
            initial_bankroll (float): Starting amount of money for betting.

        Returns:
            pd.DataFrame: DataFrame containing backtest results for each game.
        """
        print("Training the initial model...")
        self.train_model()
        print("Initial training completed.\n")

        bankroll = initial_bankroll
        results = []

        # Iterate through each game in the backtest set
        for index, game in self.backtest_data.iterrows():
            game_id = game['game_id'] if 'game_id' in game else index
            date = game['date'] if 'date' in game else game['game_date']
            home_moneyline = game[self.moneyline_columns[0]]
            away_moneyline = game[self.moneyline_columns[1]]
            actual_outcome = game[self.target_column]  # Assuming 1 = home win, 0 = away win

            # Extract feature vector for the game (exclude moneyline columns and target)
            feature_columns = [col for col in self.backtest_data.columns if col not in self.moneyline_columns + [self.target_column]]
            game_features = game[feature_columns].to_frame().T

            # Predict probability of home win
            prob_home_win = self.predict_proba_single_game(game_features)

            # Convert moneylines to decimal odds
            decimal_odds_home = self.moneyline_to_decimal(home_moneyline)
            decimal_odds_away = self.moneyline_to_decimal(away_moneyline)

            # Calculate expected value for betting on home
            expected_value_home = prob_home_win * (decimal_odds_home - 1) - (1 - prob_home_win)
            # Calculate Kelly fraction for betting on home
            kelly_fraction_home = self.kelly_criterion(prob=prob_home_win, odds=decimal_odds_home)

            # Similarly, calculate for betting on away
            prob_away_win = 1 - prob_home_win
            expected_value_away = prob_away_win * (decimal_odds_away - 1) - (1 - prob_away_win)
            kelly_fraction_away = self.kelly_criterion(prob=prob_away_win, odds=decimal_odds_away)

            # Decide which team to bet on based on higher expected value
            if expected_value_home > expected_value_away:
                bet_on = 'home'
                kelly_fraction = kelly_fraction_home
                odds = decimal_odds_home
                expected_return = expected_value_home
            else:
                bet_on = 'away'
                kelly_fraction = kelly_fraction_away
                odds = decimal_odds_away
                expected_return = expected_value_away

            # Calculate bet amount
            bet_amount = kelly_fraction * bankroll if kelly_fraction > 0 else 0

            # Calculate potential payout
            potential_payout = bet_amount * (odds - 1)

            # Determine if the bet was successful
            if bet_on == 'home':
                bet_won = 1 if actual_outcome == 1 else 0
            else:
                bet_won = 1 if actual_outcome == 0 else 0

            # Update bankroll
            if bet_won:
                bankroll += potential_payout
                profit = potential_payout
            else:
                bankroll -= bet_amount
                profit = -bet_amount

            # Record the result
            results.append({
                'game_id': game_id,
                'date': date,
                'bet_on': bet_on,
                'prob_home_win': prob_home_win,
                'prob_away_win': prob_away_win,
                'home_moneyline': home_moneyline,
                'away_moneyline': away_moneyline,
                'kelly_fraction': kelly_fraction,
                'bet_amount': bet_amount,
                'odds': odds,
                'bet_won': bet_won,
                'profit': profit,
                'bankroll': bankroll
            })

            # Update the model with the outcome of the game
            # Prepare the target for training: actual_outcome
            # Assuming that the outcome is binary: 1 = home win, 0 = away win
            self.update_model(X_new=game_features, y_new=pd.Series([actual_outcome]))

        # Convert results to DataFrame
        backtest_results = pd.DataFrame(results)
        return backtest_results

    def kelly_criterion(self, prob: float, odds: float) -> float:
        """
        Calculates the Kelly fraction for a given probability and odds.

        Parameters:
            prob (float): Estimated probability of winning (0 < prob < 1).
            odds (float): Decimal odds (odds > 1).

        Returns:
            float: Kelly fraction (0 <= fraction <= 1).
        """
        if odds <= 1:
            return 0.0
        return max((prob * (odds - 1) - (1 - prob)) / (odds - 1), 0.0)

    def moneyline_to_decimal(self, moneyline: float) -> float:
        """
        Converts American moneyline odds to decimal odds.

        Parameters:
            moneyline (float): American moneyline odds.

        Returns:
            float: Decimal odds.
        """
        if moneyline > 0:
            return (moneyline / 100) + 1
        elif moneyline < 0:
            return (100 / abs(moneyline)) + 1
        else:
            # Handle cases where moneyline is zero or invalid
            return 1.0  # Represents no payout

    def evaluate_backtest(self, backtest_results: pd.DataFrame):
        """
        Evaluates the profitability of the backtest simulation.

        Parameters:
            backtest_results (pd.DataFrame): DataFrame containing backtest results for each game.

        Returns:
            dict: Dictionary containing total profit, ROI, number of bets, number of wins, etc.
        """
        total_profit = backtest_results['profit'].sum()
        total_bet_amount = backtest_results['bet_amount'].sum()
        roi = (total_profit / total_bet_amount) * 100 if total_bet_amount > 0 else 0
        total_bets = len(backtest_results)
        total_wins = backtest_results['bet_won'].sum()
        win_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0

        evaluation = {
            'Total Profit': total_profit,
            'Return on Investment (ROI %)': roi,
            'Total Bets': total_bets,
            'Total Wins': total_wins,
            'Win Rate (%)': win_rate,
            'Final Bankroll': backtest_results['bankroll'].iloc[-1] if not backtest_results.empty else 0
        }

        return evaluation

    def run_full_pipeline(self):
        """
        Executes the full pipeline: training, backtesting, and evaluation.

        Returns:
            dict: Dictionary containing accuracy metrics, classification report, confusion matrix, and backtest evaluation.
            pd.DataFrame: DataFrame containing backtest results for each game.
        """
        # Run backtest
        print("Starting backtest simulation...")
        backtest_results = self.run_backtest()
        print("Backtest simulation completed.\n")

        # Evaluate profitability
        print("Evaluating backtest performance...")
        backtest_evaluation = self.evaluate_backtest(backtest_results)
        print("Backtest evaluation completed.\n")

        # For accuracy evaluation, evaluate on the initial training set
        print("Evaluating model accuracy on the initial training set...")
        y_train_pred = self.pipeline.predict(self.X_train)
        accuracy = accuracy_score(self.y_train, y_train_pred)
        precision = precision_score(self.y_train, y_train_pred, zero_division=0)
        recall = recall_score(self.y_train, y_train_pred, zero_division=0)
        f1 = f1_score(self.y_train, y_train_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_train, self.pipeline.predict_proba(self.X_train)[:,1])

        accuracy_metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }

        # Get classification report and confusion matrix
        classification_rep = classification_report(self.y_train, y_train_pred, zero_division=0)
        conf_matrix = confusion_matrix(self.y_train, y_train_pred)

        results = {
            'Accuracy_Metrics': accuracy_metrics,
            'Classification_Report': classification_rep,
            'Confusion_Matrix': conf_matrix,
            'Backtest_Evaluation': backtest_evaluation,
            'Backtest_Results': backtest_results
        }

        return results

    def get_trained_pipeline(self):
        """
        Returns the trained pipeline for further use or inspection.

        Returns:
            pipeline (Pipeline): Trained scikit-learn Pipeline object.
        """
        return self.pipeline

import xgboost as xgb
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

warnings.filterwarnings("ignore")  # To suppress warnings for cleaner output


class BacktestingEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        moneyline_columns: list,
        model_type: str = "logistic_regression",
        initial_train_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initializes the BacktestingEngine with the dataset and model parameters.
        """
        self.y_train = None
        self.X_train = None
        self.y = None
        self.X = None
        self.preprocessor = None
        self.data = data.copy()
        self.target_column = target_column
        self.moneyline_columns = moneyline_columns  # ['home_odds', 'away_odds']
        self.model_type = model_type.lower()
        self.initial_train_size = initial_train_size
        self.random_state = random_state

        # Initialize placeholders
        self.model = None
        self.pipeline = None
        self.initial_train_data = None
        self.backtest_data = None
        self.feature_names = None  # For feature importance

        # Preprocessing and model selection
        self.preprocess_data()
        self.split_data()
        self.select_model()

    def preprocess_data(self):
        """
        Preprocesses the data by handling missing values, encoding categorical variables,
        and scaling numerical features.
        """
        # Sort data chronologically based on 'Game_Date' or 'date' column
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
        X = self.data.drop(columns=[self.target_column])
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
        self.feature_names = self.get_feature_names()

    def get_feature_names(self):
        """
        Retrieves the feature names after preprocessing.

        Returns:
            list: List of feature names.
        """
        # Fit the preprocessor to get feature names
        self.preprocessor.fit(self.X)
        return self.preprocessor.get_feature_names_out()

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
        """
        if self.model_type == "logistic_regression":
            base_model = LogisticRegression(random_state=self.random_state)
        elif self.model_type == "xgboost":
            base_model = xgb.XGBClassifier(
                eval_metric='logloss',
                random_state=self.random_state
            )
        elif self.model_type == "random_forest":
            base_model = RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == "gradient_boosting":
            base_model = GradientBoostingClassifier(random_state=self.random_state)
        elif self.model_type == "svm":
            base_model = SVC(probability=True, random_state=self.random_state)
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported.")

        # Wrap the base model with CalibratedClassifierCV for probability calibration
        calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)

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
        Updates the model with new data by retraining.
        """
        # Append new data to training set
        self.X_train = pd.concat([self.X_train, X_new], ignore_index=True)
        self.y_train = pd.concat([self.y_train, y_new], ignore_index=True)

        # Retrain the pipeline
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
        Executes the backtesting simulation by placing one bet per day on the model's strongest favorite.
        """
        print("Training the initial model...")
        self.train_model()
        print("Initial training completed.\n")

        bankroll = initial_bankroll
        results = []

        # Ensure 'Game_Date' or 'date' column exists
        date_column = 'Game_Date' if 'Game_Date' in self.backtest_data.columns else 'date'

        # Group the backtest data by date
        grouped = self.backtest_data.groupby(date_column)

        # Iterate through each date
        for date, group in grouped:
            # Predict probabilities for all games on this date
            game_probs = []
            for index, game in group.iterrows():
                game_id = game['Game_PK']
                home_odds = game[self.moneyline_columns[0]]  # 'home_odds'
                away_odds = game[self.moneyline_columns[1]]  # 'away_odds'
                actual_outcome = game[self.target_column]  # Assuming 1 = home win, 0 = away win

                # Extract feature vector for the game (exclude moneyline columns and target)
                feature_columns = [col for col in self.backtest_data.columns if col not in self.moneyline_columns + [self.target_column]]
                game_features = game[feature_columns].to_frame().T

                # Predict probability of home win
                prob_home_win = self.predict_proba_single_game(game_features)

                # Store the game information and probability
                game_probs.append({
                    'game_id': game_id,
                    'date': date,
                    'game': game,
                    'game_features': game_features,
                    'prob_home_win': prob_home_win,
                    'actual_outcome': actual_outcome
                })

            # Select the game with the highest predicted probability (farthest from 0.5)
            highest_prob_game = max(game_probs, key=lambda x: abs(x['prob_home_win'] - 0.5))

            # Decide on which team to bet based on higher predicted probability
            prob_home_win = highest_prob_game['prob_home_win']
            if prob_home_win > 0.5:
                bet_on = 'home'
                prob = prob_home_win
                moneyline = highest_prob_game['game'][self.moneyline_columns[0]]  # home_odds
            else:
                bet_on = 'away'
                prob = 1 - prob_home_win
                moneyline = highest_prob_game['game'][self.moneyline_columns[1]]  # away_odds

            # Convert moneyline to decimal odds
            decimal_odds = self.moneyline_to_decimal(moneyline)

            # Calculate bet amount using Kelly Criterion
            kelly_fraction = self.kelly_criterion(prob=prob, odds=decimal_odds)
            bet_amount = 0.2 * kelly_fraction * bankroll if kelly_fraction > 0 else 0  # Fractional Kelly

            # Calculate potential payout
            potential_payout = bet_amount * (decimal_odds - 1)

            # Determine if the bet was successful
            actual_outcome = highest_prob_game['actual_outcome']
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
                'game_id': highest_prob_game['game_id'],
                'date': date,
                'bet_on': bet_on,
                'prob': prob,
                'moneyline': moneyline,
                'decimal_odds': decimal_odds,
                'kelly_fraction': kelly_fraction,
                'bet_amount': bet_amount,
                'potential_payout': potential_payout,
                'bet_won': bet_won,
                'profit': profit,
                'bankroll': bankroll
            })

            # Update the model with the outcome of all games on this date
            for game_prob in game_probs:
                self.update_model(X_new=game_prob['game_features'], y_new=pd.Series([game_prob['actual_outcome']]))

        # Convert results to DataFrame
        backtest_results = pd.DataFrame(results)
        return backtest_results

    @staticmethod
    def kelly_criterion(prob: float, odds: float) -> float:
        """
        Calculates the Kelly fraction for a given probability and odds.
        """
        if odds <= 1:
            return 0.0
        kelly = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
        return max(kelly, 0.0)

    @staticmethod
    def moneyline_to_decimal(moneyline: float) -> float:
        """
        Converts American moneyline odds to decimal odds.
        """
        if moneyline > 0:
            return (moneyline / 100) + 1
        elif moneyline < 0:
            return (100 / abs(moneyline)) + 1
        else:
            return 1.0  # Represents no payout

    @staticmethod
    def evaluate_backtest(backtest_results: pd.DataFrame, initial_bankroll: float):
        """
        Evaluates the profitability of the backtest simulation.
        """
        total_profit = backtest_results['profit'].sum()
        final_bankroll = backtest_results['bankroll'].iloc[-1] if not backtest_results.empty else initial_bankroll
        roi = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
        total_bets = len(backtest_results)
        total_wins = backtest_results['bet_won'].sum()
        win_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0

        evaluation = {
            'Total Profit': total_profit,
            'Return on Investment (ROI %)': roi,
            'Total Bets': total_bets,
            'Total Wins': total_wins,
            'Win Rate (%)': win_rate,
            'Final Bankroll': final_bankroll
        }

        return evaluation

    def run_full_pipeline(self, initial_bankroll: float = 10000.0):
        """
        Executes the full pipeline: training, backtesting, and evaluation.
        """
        # Run backtest
        print("Starting backtest simulation...")
        backtest_results = self.run_backtest(initial_bankroll=initial_bankroll)
        print("Backtest simulation completed.\n")

        # Evaluate profitability
        print("Evaluating backtest performance...")
        backtest_evaluation = self.evaluate_backtest(backtest_results, initial_bankroll=initial_bankroll)
        print("Backtest evaluation completed.\n")

        # For accuracy evaluation, evaluate on the initial training set
        print("Evaluating model accuracy on the initial training set...")
        y_train_pred = self.pipeline.predict(self.X_train)
        accuracy = accuracy_score(self.y_train, y_train_pred)
        precision = precision_score(self.y_train, y_train_pred, zero_division=0)
        recall = recall_score(self.y_train, y_train_pred, zero_division=0)
        f1 = f1_score(self.y_train, y_train_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_train, self.pipeline.predict_proba(self.X_train)[:, 1])

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

        # Get feature importances
        feature_importances = self.get_feature_importances()

        results = {
            'Accuracy_Metrics': accuracy_metrics,
            'Classification_Report': classification_rep,
            'Confusion_Matrix': conf_matrix,
            'Backtest_Evaluation': backtest_evaluation,
            'Backtest_Results': backtest_results,
            'Feature_Importances': feature_importances
        }

        return results

    def get_feature_importances(self):
        """
        Retrieves feature importances from the trained model.

        Returns:
            pd.DataFrame: DataFrame containing feature names and their importances.
        """
        # Access the calibrated classifier
        calibrated_classifier = self.pipeline.named_steps['calibrated_classifier']
        # List to store importances from each fitted estimator
        importances_list = []
        feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()

        # Handle different model types
        if self.model_type in ["random_forest", "gradient_boosting", "xgboost"]:
            # For tree-based models, extract feature importances from each fitted estimator
            for calibrated_clf in calibrated_classifier.calibrated_classifiers_:
                estimator = calibrated_clf.estimator  # Corrected attribute
                importances_list.append(estimator.feature_importances_)
            # Average the importances
            importances = np.mean(importances_list, axis=0)
        elif self.model_type == "logistic_regression":
            # For logistic regression, use coefficients
            for calibrated_clf in calibrated_classifier.calibrated_classifiers_:
                estimator = calibrated_clf.estimator  # Corrected attribute
                importances_list.append(np.abs(estimator.coef_[0]))
            # Average the importances
            importances = np.mean(importances_list, axis=0)
        else:
            # For models without direct feature importances
            print(f"Feature importances not available for model type '{self.model_type}'.")
            return None

        # Create a DataFrame for feature importances
        feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Sort features by importance
        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
        feature_importances.reset_index(drop=True, inplace=True)

        print("\nFeature Importances:")
        print(feature_importances.head(10))  # Display top 10 features

        return feature_importances

    def get_trained_pipeline(self):
        """
        Returns the trained pipeline for further use or inspection.
        """
        return self.pipeline

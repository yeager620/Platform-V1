import xgboost as xgb
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
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
    brier_score_loss,
    log_loss,
    roc_curve
)
from sklearn.calibration import calibration_curve
from scipy.stats import ttest_1samp, wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings("ignore")


class BacktestingEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        moneyline_columns: list,
        model_type: str = "logistic_regression",
        initial_train_size: float = 0.2,
        kelly_fraction: float = 1.0,  # Full Kelly by default
        output_folder: str = "/Users/yeager/Desktop/Maquoketa-Platform-V1/x-backtests/automated-reports",
        random_state: int = 42,
    ):
        """
        Initializes the BacktestingEngine with the dataset and model parameters.

        Parameters:
            data (pd.DataFrame): The dataset containing historical game data.
            target_column (str): The name of the target column indicating the game outcome.
            moneyline_columns (list): List containing the names of the home and away moneyline columns.
            model_type (str): The type of model to use ('logistic_regression', 'xgboost', etc.).
            initial_train_size (float): The fraction of data to use for initial training.
            kelly_fraction (float): Fraction of the Kelly Criterion to use for bet sizing.
            output_folder (str): The folder where the test report will be saved.
            random_state (int): Random state for reproducibility.
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
        self.kelly_fraction = kelly_fraction  # Fractional Kelly (e.g., 0.2 for 1/5 Kelly)
        self.output_folder = output_folder
        self.random_state = random_state

        # Initialize placeholders
        self.model = None
        self.pipeline = None
        self.initial_train_data = None
        self.backtest_data = None
        self.feature_names = None  # For feature importance

        # New placeholders for comparison
        self.bookmaker_probs = []
        self.model_probs = []

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

    def adjust_bookmaker_probabilities(self, home_odds, away_odds):
        """
        Adjusts the bookmakers' implied probabilities to account for the overround.

        Parameters:
            home_odds (float): Moneyline odds for the home team.
            away_odds (float): Moneyline odds for the away team.

        Returns:
            tuple: Adjusted probabilities for home win and away win.
        """
        # Convert moneyline odds to decimal odds
        decimal_home_odds = self.moneyline_to_decimal(home_odds)
        decimal_away_odds = self.moneyline_to_decimal(away_odds)

        # Calculate implied probabilities
        implied_prob_home = 1 / decimal_home_odds
        implied_prob_away = 1 / decimal_away_odds

        # Sum of implied probabilities (overround)
        sum_implied_probs = implied_prob_home + implied_prob_away

        # Adjust probabilities
        adjusted_prob_home = implied_prob_home / sum_implied_probs
        adjusted_prob_away = implied_prob_away / sum_implied_probs

        return adjusted_prob_home, adjusted_prob_away

    def run_backtest(self, initial_bankroll: float = 10000.0):
        """
        Executes the backtesting simulation by placing one bet per day on the model's strongest favorite.
        Also collects probabilities for comparison with bookmakers.
        """
        print("Training the initial model...")
        self.train_model()
        print("Initial training completed.\n")

        bankroll = initial_bankroll
        results = []

        # Lists to store predictions and actual outcomes
        backtest_predictions = []
        backtest_actuals = []
        backtest_profits = []  # To store individual bet profits/losses

        # For probability comparison
        model_probabilities = []
        bookmaker_probabilities = []
        actual_outcomes = []

        # Ensure 'Game_Date' or 'date' column exists
        date_column = 'Game_Date' if 'Game_Date' in self.backtest_data.columns else 'date'

        # Group the backtest data by date
        grouped = self.backtest_data.groupby(date_column)

        # Iterate through each date
        for date, group in tqdm(grouped, total=len(grouped), desc="Processing Dates"):
            # Predict probabilities for all games on this date
            game_probs = []
            for index, game in group.iterrows():
                game_id = game['Game_PK']
                home_odds = game[self.moneyline_columns[0]]  # 'home_odds'
                away_odds = game[self.moneyline_columns[1]]  # 'away_odds'
                actual_outcome = game[self.target_column]  # Assuming 1 = home win, 0 = away win

                # Extract feature vector for the game (exclude moneyline columns and target)
                feature_columns = [col for col in self.backtest_data.columns if
                                   col not in self.moneyline_columns + [self.target_column]]
                game_features = game[feature_columns].to_frame().T

                # Predict probability of home win
                prob_home_win = self.predict_proba_single_game(game_features)

                # Adjust bookmakers' probabilities
                adjusted_prob_home, adjusted_prob_away = self.adjust_bookmaker_probabilities(home_odds, away_odds)

                # Store probabilities for comparison
                model_probabilities.append(prob_home_win)
                bookmaker_probabilities.append(adjusted_prob_home)
                actual_outcomes.append(actual_outcome)

                # Store the game information and probability
                game_probs.append({
                    'game_id': game_id,
                    'date': date,
                    'game': game,
                    'game_features': game_features,
                    'prob_home_win': prob_home_win,
                    'adjusted_prob_home': adjusted_prob_home,
                    'actual_outcome': actual_outcome
                })

            # Select the game with the highest predicted probability difference from bookmakers
            highest_prob_game = max(
                game_probs,
                key=lambda x: abs(x['prob_home_win'] - x['adjusted_prob_home'])
            )

            # Decide on which team to bet based on higher predicted probability
            prob_home_win = highest_prob_game['prob_home_win']
            adjusted_prob_home = highest_prob_game['adjusted_prob_home']
            if prob_home_win > 0.5:
                bet_on = 'home'
                prob = prob_home_win
                moneyline = highest_prob_game['game'][self.moneyline_columns[0]]  # home_odds
                predicted_label = 1  # Home win
            else:
                bet_on = 'away'
                prob = 1 - prob_home_win
                moneyline = highest_prob_game['game'][self.moneyline_columns[1]]  # away_odds
                predicted_label = 0  # Away win

            # Collect predictions and actual outcomes
            backtest_predictions.append(predicted_label)
            backtest_actuals.append(highest_prob_game['actual_outcome'])

            # Convert moneyline to decimal odds
            decimal_odds = self.moneyline_to_decimal(moneyline)

            # Calculate bet amount using Kelly Criterion
            f_opt = self.kelly_criterion(prob=prob, odds=decimal_odds)
            bet_amount = self.kelly_fraction * f_opt * bankroll if f_opt > 0 else 0  # Fractional Kelly

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

            # Collect individual profits/losses
            backtest_profits.append(profit)

            # Record the result
            results.append({
                'game_id': highest_prob_game['game_id'],
                'date': date,
                'bet_on': bet_on,
                'prob': prob,
                'moneyline': moneyline,
                'decimal_odds': decimal_odds,
                'full_kelly_fraction': f_opt,
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

        # Store backtest predictions and actuals
        self.backtest_predictions = backtest_predictions
        self.backtest_actuals = backtest_actuals
        self.backtest_profits = backtest_profits  # Store profits for statistical tests

        # Store probabilities for comparison
        self.model_probs = model_probabilities
        self.bookmaker_probs = bookmaker_probabilities
        self.actual_outcomes = actual_outcomes

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

    def evaluate_backtest(self, backtest_results: pd.DataFrame, initial_bankroll: float):
        """
        Evaluates the profitability of the backtest simulation and performs statistical tests.
        Also compares model probabilities with bookmakers' probabilities.
        """
        total_profit = backtest_results['profit'].sum()
        final_bankroll = backtest_results['bankroll'].iloc[-1] if not backtest_results.empty else initial_bankroll
        roi = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
        total_bets = len(backtest_results)
        total_wins = backtest_results['bet_won'].sum()
        win_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0

        # Perform statistical tests on profits
        profits = np.array(self.backtest_profits)
        # Remove zero profits (in case of no bets placed)
        profits = profits[profits != 0]

        # One-sample t-test (testing if mean profit is greater than zero)
        t_stat, t_p_value = ttest_1samp(profits, popmean=0, alternative='greater')

        # Wilcoxon Signed-Rank Test (non-parametric test)
        try:
            w_stat, w_p_value = wilcoxon(profits - 0, alternative='greater')
        except ValueError:
            # If all profits are the same value, the test cannot be performed
            w_stat, w_p_value = np.nan, np.nan

        # Mann-Whitney U-test (non-parametric test)
        try:
            # Create a zero-profit array for comparison
            zero_profits = np.zeros_like(profits)
            u_stat, u_p_value = mannwhitneyu(profits, zero_profits, alternative='greater')
        except ValueError:
            u_stat, u_p_value = np.nan, np.nan

        # Calculate Brier Scores
        model_brier = brier_score_loss(self.actual_outcomes, self.model_probs)
        bookmaker_brier = brier_score_loss(self.actual_outcomes, self.bookmaker_probs)

        # Calculate Log Loss
        model_log_loss = log_loss(self.actual_outcomes, self.model_probs)
        bookmaker_log_loss = log_loss(self.actual_outcomes, self.bookmaker_probs)

        # Calculate AUC
        model_auc = roc_auc_score(self.actual_outcomes, self.model_probs)
        bookmaker_auc = roc_auc_score(self.actual_outcomes, self.bookmaker_probs)

        # Statistical test on Brier Scores (Diebold-Mariano Test)
        loss_diff = np.array([(mp - ao) ** 2 - (bp - ao) ** 2 for mp, bp, ao in
                              zip(self.model_probs, self.bookmaker_probs, self.actual_outcomes)])
        dm_stat = self.diebold_mariano(loss_diff)
        p_value_dm = 1 - self.norm_cdf(dm_stat)

        evaluation = {
            'Total Profit': total_profit,
            'Return on Investment (ROI %)': roi,
            'Total Bets': total_bets,
            'Total Wins': total_wins,
            'Win Rate (%)': win_rate,
            'Final Bankroll': final_bankroll,
            'T-Test Statistic': t_stat,
            'T-Test p-value': t_p_value,
            'Wilcoxon Test Statistic': w_stat,
            'Wilcoxon Test p-value': w_p_value,
            'Mann-Whitney U Statistic': u_stat,
            'Mann-Whitney U p-value': u_p_value,
            'Model Brier Score': model_brier,
            'Bookmaker Brier Score': bookmaker_brier,
            'Model Log Loss': model_log_loss,
            'Bookmaker Log Loss': bookmaker_log_loss,
            'Model AUC': model_auc,
            'Bookmaker AUC': bookmaker_auc,
            'Diebold-Mariano Statistic': dm_stat,
            'Diebold-Mariano p-value': p_value_dm
        }

        # Store evaluation metrics for report
        self.backtest_evaluation = evaluation

        return evaluation

    @staticmethod
    def diebold_mariano(loss_diff):
        """
        Computes the Diebold-Mariano test statistic.

        Parameters:
            loss_diff (array): Loss differentials between two models.

        Returns:
            float: Diebold-Mariano test statistic.
        """
        T = len(loss_diff)
        mean_ld = np.mean(loss_diff)
        var_ld = np.var(loss_diff, ddof=1)
        dm_stat = mean_ld / np.sqrt((var_ld / T))
        return dm_stat

    @staticmethod
    def norm_cdf(value):
        """
        Computes the cumulative distribution function for a standard normal distribution.

        Parameters:
            value (float): The value to compute the CDF for.

        Returns:
            float: The CDF value.
        """
        from scipy.stats import norm
        return norm.cdf(value)

    def plot_calibration_curve(self):
        """
        Plots the calibration curves for both the model and bookmakers.
        """
        plt.figure(figsize=(10, 5))

        # Model Calibration
        prob_true_model, prob_pred_model = calibration_curve(self.actual_outcomes, self.model_probs, n_bins=10)
        plt.plot(prob_pred_model, prob_true_model, marker='o', label='Model')

        # Bookmaker Calibration
        prob_true_book, prob_pred_book = calibration_curve(self.actual_outcomes, self.bookmaker_probs, n_bins=10)
        plt.plot(prob_pred_book, prob_true_book, marker='s', label='Bookmaker')

        # Perfect Calibration Line
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

        plt.title('Calibration Curves')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.legend()
        plt.grid(True)

        # Save the plot
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        calibration_plot_file = os.path.join(self.output_folder, f'calibration_plot_{timestamp}.png')
        plt.savefig(calibration_plot_file)
        plt.close()
        print(f"Calibration plot saved to {calibration_plot_file}")

    def run_full_pipeline(self, initial_bankroll: float = 10000.0):
        """
        Executes the full pipeline: training, backtesting, evaluation, and report generation.
        """
        # Run backtest
        print("Starting backtest simulation...")
        backtest_results = self.run_backtest(initial_bankroll=initial_bankroll)
        print("Backtest simulation completed.\n")

        # Evaluate profitability
        print("Evaluating backtest performance...")
        backtest_evaluation = self.evaluate_backtest(backtest_results, initial_bankroll=initial_bankroll)
        print("Backtest evaluation completed.\n")

        # Evaluate model accuracy on backtesting predictions
        print("Evaluating model accuracy on backtesting predictions...")
        accuracy = accuracy_score(self.backtest_actuals, self.backtest_predictions)
        precision = precision_score(self.backtest_actuals, self.backtest_predictions, zero_division=0)
        recall = recall_score(self.backtest_actuals, self.backtest_predictions, zero_division=0)
        f1 = f1_score(self.backtest_actuals, self.backtest_predictions, zero_division=0)
        roc_auc = roc_auc_score(self.backtest_actuals, self.backtest_predictions)

        accuracy_metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }

        # Get classification report and confusion matrix
        classification_rep = classification_report(self.backtest_actuals, self.backtest_predictions, zero_division=0)
        conf_matrix = confusion_matrix(self.backtest_actuals, self.backtest_predictions)

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

        # Generate calibration plot
        self.plot_calibration_curve()

        # Generate and save the report
        self.generate_report(results)

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
                estimator = calibrated_clf.estimator
                if hasattr(estimator, 'feature_importances_'):
                    importances_list.append(estimator.feature_importances_)
                else:
                    print(f"Estimator does not have feature_importances_ attribute.")
                    return None
            # Average the importances
            importances = np.mean(importances_list, axis=0)
        elif self.model_type == "logistic_regression":
            # For logistic regression, use coefficients
            for calibrated_clf in calibrated_classifier.calibrated_classifiers_:
                estimator = calibrated_clf.estimator
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

    def generate_report(self, results):
        """
        Generates a test report and saves it to the output folder with a unique and descriptive file name.

        Parameters:
            results (dict): Dictionary containing evaluation metrics and backtest results.
        """
        print("Generating test report...")

        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Generate a unique and descriptive file name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        kelly_str = f"kelly_{self.kelly_fraction}"
        report_file_name = f"backtest_report_{self.model_type}_{kelly_str}_{timestamp}.txt"
        report_file = os.path.join(self.output_folder, report_file_name)

        with open(report_file, 'w') as f:
            f.write("Backtesting Report\n")
            f.write("==================\n\n")

            # Dataset Information
            total_games = len(self.data)
            date_column = 'Game_Date' if 'Game_Date' in self.data.columns else 'date'
            total_days = self.data[date_column].nunique()
            f.write(f"Total number of games in dataset: {total_games}\n")
            f.write(f"Total number of days in dataset: {total_days}\n")
            f.write(f"Initial training size: {self.initial_train_size * 100}%\n\n")

            # Kelly Fraction Information
            kelly_info = f"Kelly fraction used: {self.kelly_fraction} (Full Kelly)\n" if self.kelly_fraction == 1.0 else f"Kelly fraction used: {self.kelly_fraction} (Fractional Kelly)\n"
            f.write(kelly_info)

            # Backtest Evaluation Metrics
            f.write("\nBacktest Evaluation Metrics:\n")
            for metric, value in results['Backtest_Evaluation'].items():
                f.write(f"{metric}: {value}\n")

            # Model Accuracy Metrics
            f.write("\nModel Accuracy Metrics:\n")
            for metric, value in results['Accuracy_Metrics'].items():
                f.write(f"{metric}: {value}\n")

            # Classification Report
            f.write("\nClassification Report:\n")
            f.write(f"{results['Classification_Report']}\n")

            # Confusion Matrix
            f.write("Confusion Matrix:\n")
            f.write(f"{results['Confusion_Matrix']}\n")

            # Feature Importances
            if results['Feature_Importances'] is not None:
                f.write("\nTop 10 Feature Importances:\n")
                f.write(results['Feature_Importances'].head(10).to_string(index=False))
            else:
                f.write("\nFeature importances not available for the selected model.\n")

            # Additional Information
            f.write("\n\nAdditional Information:\n")
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Random State: {self.random_state}\n")
            f.write(f"Report Generated on: {timestamp}\n")

            # Note about Calibration Plot
            f.write("\nCalibration plot saved as 'calibration_plot_v1.1_n2849_23-24.png' in the output folder.\n")

        print(f"Test report saved to {report_file}")

    def get_trained_pipeline(self):
        """
        Returns the trained pipeline for further use or inspection.
        """
        return self.pipeline
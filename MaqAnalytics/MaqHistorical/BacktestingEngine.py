import xgboost as xgb
import optuna
import pandas as pd
import numpy as np
import os
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from scipy.stats import ttest_1samp, wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
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
            update_model: bool = True,
            output_folder: str = "/Users/yeager/Desktop/Maquoketa-Platform-V1/x-backtests/automated-reports",
            random_state: int = 42,
    ):
        self.data = data.copy()

        self.target_column = target_column
        self.run_diff_col = "Run_Diff"
        self.moneyline_columns = moneyline_columns

        self.model_type = model_type.lower()
        self.initial_train_size = initial_train_size
        self.kelly_fraction = kelly_fraction
        self.output_folder = output_folder
        self.random_state = random_state
        self.update_model_daily = update_model

        # Initialize placeholders
        self.model = None
        self.pipeline = None
        self.initial_train_data = None
        self.backtest_data = None
        self.feature_names = None  # For feature importance
        self.best_params = None

        # Initialize placeholders for probabilities and actual outcomes
        self.all_model_probs = []
        self.all_actual_outcomes = []
        self.all_bookmaker_probs = []

        # Preprocessing and model selection
        self.preprocess_data()
        self.split_data()
        self.select_model()

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
        self.feature_names = self.get_feature_names()

    def get_feature_names(self):
        self.preprocessor.fit(self.X)
        return self.preprocessor.get_feature_names_out()

    def split_data(self):
        total_games = len(self.data)
        initial_train_end = int(total_games * self.initial_train_size)
        # initial_train_start = int(total_games * 0.25)

        self.initial_train_data = self.data.iloc[:initial_train_end].copy()
        self.backtest_data = self.data.iloc[initial_train_end:].copy()

        # Separate features and target for initial training
        X_train = self.initial_train_data.drop(columns=[self.target_column, "Game_Date", "Game_PK"])
        y_train = self.initial_train_data[self.target_column]

        self.X_train = X_train
        self.y_train = y_train

    def select_model(self):
        if self.model_type == "logistic_regression":
            base_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
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
        # calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)

        # Create a pipeline that first preprocesses the data and then fits the calibrated model
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", base_model)
        ])

    def train_model(self):
        self.pipeline.fit(self.X_train, self.y_train)

    def tune_hyperparameters(self):
        # Define parameter grids for different model types
        # Adjust these to your liking
        if self.model_type == "logistic_regression":
            param_grid = {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs"]
            }
        elif self.model_type == "xgboost":
            param_grid = {
                "classifier__n_estimators": [50, 75, 100],
                "classifier__max_depth": [2, 3],
                "classifier__learning_rate": [0.075, 0.1, 0.2]
            }
        elif self.model_type == "random_forest":
            param_grid = {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__max_features": ["sqrt", "log2"]
            }
        elif self.model_type == "gradient_boosting":
            param_grid = {
                "classifier__n_estimators": [50, 100],
                "classifier__learning_rate": [0.01, 0.1],
                "classifier__max_depth": [3, 5]
            }
        elif self.model_type == "svm":
            param_grid = {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["linear", "rbf"]
            }
        else:
            # If no grid defined, skip tuning
            return

        print("Starting hyperparameter tuning on initial training data...")
        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=6,  # n-fold cross-validation
            scoring='neg_log_loss',
            n_jobs=--1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        self.best_params = grid_search.best_params_

        print("Best hyperparameters found:", grid_search.best_params_)

        # Update pipeline with best parameters
        self.pipeline = grid_search.best_estimator_

        # Now wrap the best found classifier with CalibratedClassifierCV
        best_classifier = self.pipeline.named_steps['classifier']
        calibrated_model = CalibratedClassifierCV(estimator=best_classifier, method='sigmoid', cv=10, n_jobs=-1)

        # Rebuild pipeline with calibration
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("calibrated_classifier", calibrated_model)
        ])

        # Fit once to calibrate with the initial training set
        self.pipeline.fit(self.X_train, self.y_train)
        print("Hyperparameter tuning and calibration complete.\n")

    def update_model(self, X_new, y_new):
        self.X_train = pd.concat([self.X_train, X_new], ignore_index=True)
        self.y_train = pd.concat([self.y_train, y_new], ignore_index=True)

        self.pipeline.fit(self.X_train, self.y_train)

    def predict_proba_single_game(self, game_features):
        proba = self.pipeline.predict_proba(game_features)[0][1]  # Assuming positive class is home win
        return proba

    def adjust_bookmaker_probabilities(self, home_odds, away_odds):
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
        print("Training the initial model...")

        # self.train_model()

        self.tune_hyperparameters()

        print("Initial training completed.\n")

        bankroll = initial_bankroll
        results = []

        # Lists to store predictions and actual outcomes for betting
        backtest_predictions = []
        backtest_actuals = []
        backtest_profits = []  # To store individual bet profits/losses

        # For probabilities and actual outcomes for all games
        all_model_probs = []
        all_actual_outcomes = []
        all_bookmaker_probs = []

        # Precompute feature columns outside the loop
        excluded_columns = self.moneyline_columns + [self.target_column, 'Game_Date', 'Game_PK']
        feature_columns = [col for col in self.backtest_data.columns if col not in excluded_columns]

        # Group the backtest data by date
        date_column = 'Game_Date' if 'Game_Date' in self.backtest_data.columns else 'date'
        grouped = self.backtest_data.groupby(date_column)

        # Iterate through each date
        for date, group in tqdm(grouped, total=grouped.ngroups, desc="Processing Dates"):
            # Extract all feature vectors for the current group
            game_features_df = group[feature_columns]
            home_odds = group[self.moneyline_columns[0]].values  # e.g., 'home_odds'
            away_odds = group[self.moneyline_columns[1]].values  # e.g., 'away_odds'
            actual_outcomes = group[self.target_column].values  # 1 = home win, 0 = away win
            game_ids = group['Game_PK'].values

            # Batch predict probabilities
            prob_home_win = self.pipeline.predict_proba(game_features_df)[:, 1]  # Probability of home win

            # Adjust bookmaker probabilities vectorially
            adjusted_prob_home, _ = self.adjust_bookmaker_probabilities(home_odds, away_odds)

            # Collect all required data
            all_model_probs.extend(prob_home_win)
            all_actual_outcomes.extend(actual_outcomes)
            all_bookmaker_probs.extend(adjusted_prob_home)

            # Calculate the absolute difference between model and bookmaker probabilities
            prob_diff = np.abs(prob_home_win - adjusted_prob_home)

            # Identify the index of the game with the highest probability difference
            best_game_idx = np.argmax(prob_diff)

            # Extract the best game's details
            best_game_id = game_ids[best_game_idx]
            best_game_prob = prob_home_win[best_game_idx]
            best_game_bookmaker_prob = adjusted_prob_home[best_game_idx]
            best_game_actual_outcome = actual_outcomes[best_game_idx]
            best_game_row = group.iloc[best_game_idx]

            # Decide on which team to bet based on higher predicted probability
            if best_game_prob > 0.5:
                bet_on = 'home'
                prob = best_game_prob
                moneyline = best_game_row[self.moneyline_columns[0]]  # home_odds
                predicted_label = 1  # Home win
            else:
                bet_on = 'away'
                prob = 1 - best_game_prob
                moneyline = best_game_row[self.moneyline_columns[1]]  # away_odds
                predicted_label = 0  # Away win

            # Collect predictions and actual outcomes for bets
            backtest_predictions.append(predicted_label)
            backtest_actuals.append(best_game_actual_outcome)

            # Convert moneyline to decimal odds
            decimal_odds = self.moneyline_to_decimal(moneyline)

            # Calculate bet amount using Kelly Criterion
            f_opt = self.kelly_criterion(prob=prob, odds=decimal_odds)
            bet_amount = self.kelly_fraction * f_opt * bankroll if f_opt > 0 else 0  # Fractional Kelly

            # Calculate potential payout
            potential_payout = bet_amount * (decimal_odds - 1)

            # Determine if the bet was successful
            if bet_on == 'home':
                bet_won = 1 if best_game_actual_outcome == 1 else 0
            else:
                bet_won = 1 if best_game_actual_outcome == 0 else 0

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
                'game_id': best_game_id,
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

            # Prepare data for model update
            # Extract features and outcomes for all games in the current group
            X_new = game_features_df
            y_new = pd.Series(actual_outcomes)

            # Update the model with the outcomes of all games on this date
            if self.update_model_daily:
                self.update_model(X_new=X_new, y_new=y_new)

        # Store probabilities and actual outcomes for all games
        self.all_model_probs = all_model_probs
        self.all_actual_outcomes = all_actual_outcomes
        self.all_bookmaker_probs = all_bookmaker_probs

        # Convert results to DataFrame
        backtest_results = pd.DataFrame(results)

        # Store backtest predictions and actuals
        self.backtest_predictions = backtest_predictions
        self.backtest_actuals = backtest_actuals
        self.backtest_profits = backtest_profits

        return backtest_results

    @staticmethod
    def kelly_criterion(prob: float, odds: float) -> float:
        if odds <= 1:
            return 0.0
        kelly = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
        return max(kelly, 0.0)

    @staticmethod
    def moneyline_to_decimal(moneyline):
        """
        Convert moneyline odds to decimal odds.

        Parameters:
        - moneyline (float or array-like): The moneyline odds. Positive for underdogs, negative for favorites.

        Returns:
        - float or np.ndarray: Decimal odds corresponding to the input moneyline odds.
        """
        moneyline = np.asarray(moneyline)
        decimal_odds = np.where(
            moneyline > 0,
            (moneyline / 100) + 1,
            np.where(
                moneyline < 0,
                (100 / np.abs(moneyline)) + 1,
                1.0  # Represents no payout
            )
        )
        return decimal_odds

    def evaluate_backtest(self, backtest_results: pd.DataFrame, initial_bankroll: float):
        # Profitability evaluation code remains the same
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
            w_stat, w_p_value = np.nan, np.nan

        # Mann-Whitney U-test (non-parametric test)
        try:
            zero_profits = np.zeros_like(profits)
            u_stat, u_p_value = mannwhitneyu(profits, zero_profits, alternative='greater')
        except ValueError:
            u_stat, u_p_value = np.nan, np.nan

        # Calculate evaluation metrics using all probabilities and actual outcomes
        model_brier = brier_score_loss(self.all_actual_outcomes, self.all_model_probs)
        bookmaker_brier = brier_score_loss(self.all_actual_outcomes, self.all_bookmaker_probs)

        model_log_loss = log_loss(self.all_actual_outcomes, self.all_model_probs)
        bookmaker_log_loss = log_loss(self.all_actual_outcomes, self.all_bookmaker_probs)

        model_auc = roc_auc_score(self.all_actual_outcomes, self.all_model_probs)
        bookmaker_auc = roc_auc_score(self.all_actual_outcomes, self.all_bookmaker_probs)

        # Statistical test on Brier Scores (Diebold-Mariano Test)
        loss_diff = np.array([(mp - ao) ** 2 - (bp - ao) ** 2 for mp, bp, ao in
                              zip(self.all_model_probs, self.all_bookmaker_probs, self.all_actual_outcomes)])
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

        self.backtest_evaluation = evaluation

        return evaluation

    @staticmethod
    def diebold_mariano(loss_diff):
        T = len(loss_diff)
        mean_ld = np.mean(loss_diff)
        var_ld = np.var(loss_diff, ddof=1)
        dm_stat = mean_ld / np.sqrt((var_ld / T))
        return dm_stat

    @staticmethod
    def norm_cdf(value):
        from scipy.stats import norm
        return norm.cdf(value)

    def plot_calibration_curve(self):
        plt.figure(figsize=(10, 5))

        # Model Calibration
        prob_pred_model, prob_true_model = calibration_curve(self.all_actual_outcomes, self.all_model_probs, n_bins=10)
        plt.plot(prob_pred_model, prob_true_model, marker='o', label='Model')

        # Bookmaker Calibration
        prob_pred_book, prob_true_book = calibration_curve(self.all_actual_outcomes, self.all_bookmaker_probs,
                                                           n_bins=10)
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
        print("Starting backtest simulation...")
        backtest_results = self.run_backtest(initial_bankroll=initial_bankroll)
        print("Backtest simulation completed.\n")

        print("Evaluating backtest performance...")
        backtest_evaluation = self.evaluate_backtest(backtest_results, initial_bankroll=initial_bankroll)
        print("Backtest evaluation completed.\n")

        print("Evaluating model accuracy on all predictions...")
        # Convert probabilities to class labels (threshold at 0.5)
        all_predicted_labels = [1 if prob >= 0.5 else 0 for prob in self.all_model_probs]
        accuracy = accuracy_score(self.all_actual_outcomes, all_predicted_labels)
        precision = precision_score(self.all_actual_outcomes, all_predicted_labels, zero_division=0)
        recall = recall_score(self.all_actual_outcomes, all_predicted_labels, zero_division=0)
        f1 = f1_score(self.all_actual_outcomes, all_predicted_labels, zero_division=0)
        roc_auc = roc_auc_score(self.all_actual_outcomes, self.all_model_probs)

        accuracy_metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }

        # Get classification report and confusion matrix
        classification_rep = classification_report(self.all_actual_outcomes, all_predicted_labels, zero_division=0)
        conf_matrix = confusion_matrix(self.all_actual_outcomes, all_predicted_labels)

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

        # Plot bankroll over time
        self.plot_bankroll_over_time(backtest_results)

        return results

    def get_feature_importances(self):
        # Feature importance code remains the same
        calibrated_classifier = self.pipeline.named_steps['calibrated_classifier']
        importances_list = []
        feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()

        if self.model_type in ["random_forest", "gradient_boosting", "xgboost"]:
            for calibrated_clf in calibrated_classifier.calibrated_classifiers_:
                estimator = calibrated_clf.estimator
                if hasattr(estimator, 'feature_importances_'):
                    importances_list.append(estimator.feature_importances_)
                else:
                    print(f"Estimator does not have feature_importances_ attribute.")
                    return None
            importances = np.mean(importances_list, axis=0)
        elif self.model_type == "logistic_regression":
            for calibrated_clf in calibrated_classifier.calibrated_classifiers_:
                estimator = calibrated_clf.estimator
                importances_list.append(np.abs(estimator.coef_[0]))
            importances = np.mean(importances_list, axis=0)
        else:
            print(f"Feature importances not available for model type '{self.model_type}'.")
            return None

        feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
        feature_importances.reset_index(drop=True, inplace=True)

        print("\nFeature Importances:")
        print(feature_importances.head(20))  # Display top 20 features

        return feature_importances

    def generate_report(self, results):
        print("Generating test report...")

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        kelly_str = f"kelly_{self.kelly_fraction}"
        report_file_name = f"backtest_report_{self.model_type}_{kelly_str}_{timestamp}.txt"
        report_file = os.path.join(self.output_folder, report_file_name)

        with open(report_file, 'w') as f:
            f.write("Backtesting Report\n")
            f.write("==================\n\n")

            total_games = len(self.data)
            date_column = 'Game_Date' if 'Game_Date' in self.data.columns else 'date'
            total_days = self.data[date_column].nunique()
            f.write(f"Total number of games in dataset: {total_games}\n")
            f.write(f"Total number of days in dataset: {total_days}\n")
            f.write(f"Initial training size: {self.initial_train_size * 100}%\n")
            f.write(f"Best hyperparameters found: {self.best_params}\n")
            f.write(f"Daily model updates: {self.update_model_daily}\n\n")

            kelly_info = f"Kelly fraction used: {self.kelly_fraction} (Full Kelly)\n" if self.kelly_fraction == 1.0 else f"Kelly fraction used: {self.kelly_fraction} (Fractional Kelly)\n"
            f.write(kelly_info)

            f.write("\nBacktest Evaluation Metrics:\n")
            for metric, value in results['Backtest_Evaluation'].items():
                f.write(f"{metric}: {value}\n")

            f.write("\nModel Accuracy Metrics:\n")
            for metric, value in results['Accuracy_Metrics'].items():
                f.write(f"{metric}: {value}\n")

            f.write("\nClassification Report:\n")
            f.write(f"{results['Classification_Report']}\n")

            f.write("Confusion Matrix:\n")
            f.write(f"{results['Confusion_Matrix']}\n")

            if results['Feature_Importances'] is not None:
                f.write("\nTop 20 Feature Importances:\n")
                f.write(results['Feature_Importances'].head(20).to_string(index=False))
            else:
                f.write("\nFeature importances not available for the selected model.\n")

            f.write("\n\nAdditional Information:\n")
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Random State: {self.random_state}\n")
            f.write(f"Report Generated on: {timestamp}\n")

            f.write(f"\nCalibration plot saved as 'calibration_plot_{timestamp}.png' in the output folder.\n")

        print(f"Test report saved to {report_file}")

    def plot_bankroll_over_time(self, backtest_results: pd.DataFrame):
        """
        Plot bankroll over time with a log-scaled y-axis.

        :param backtest_results: DataFrame containing backtest results, including bankroll per game.
        """
        if backtest_results.empty:
            print("No results to plot.")
            return

        try:
            # Extract dates and bankroll
            dates = pd.to_datetime(backtest_results['date'])
            bankroll = backtest_results['bankroll']

            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(dates, bankroll, label='Bankroll', marker='o', markersize=4, linewidth=2)
            plt.yscale('log')  # Log scale for y-axis
            plt.title('Bankroll Over Time (Log-Scaled)')
            plt.xlabel('Date')
            plt.ylabel('Bankroll (Log Scale)')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()

            # Save the plot
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plot_file = os.path.join(self.output_folder, f'bankroll_over_time_{timestamp}.png')
            plt.savefig(plot_file)
            plt.close()

            print(f"Bankroll over time plot saved to {plot_file}")
        except Exception as e:
            print(f"Error plotting bankroll over time: {e}")

    def get_trained_pipeline(self):
        return self.pipeline

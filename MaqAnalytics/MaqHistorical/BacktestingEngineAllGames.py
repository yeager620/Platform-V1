import xgboost as xgb
import optuna
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

from sklearn.impute import KNNImputer, MissingIndicator
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
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
            update_model: bool = False,
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

        # Perform data preprocessing (but do not fit the imputer yet)
        self._prepare_data()

        # Split into initial train vs. backtest
        self.split_data()

        # Build pipeline with chosen model (KNNImputer is inside the pipeline but not yet fitted)
        self.select_model()

    def _prepare_data(self):
        """Prepare dataset by sorting, dropping NAs, etc.
           Do not fit the pipeline here to avoid lookahead bias.
        """
        if 'Game_Date' in self.data.columns:
            self.data.sort_values(by='Game_Date', inplace=True)
        else:
            raise ValueError("Data must contain a 'Game_Date' column for chronological sorting.")

        # Ensure 'park_id' is treated as a categorical variable by converting it to string
        if 'park_id' in self.data.columns:
            self.data['park_id'] = self.data['park_id'].astype(str)

        # Drop rows with missing moneyline values
        self.data.dropna(subset=self.moneyline_columns, inplace=True)

    def split_data(self):
        """Split the data chronologically into initial training set and backtest set."""
        total_games = len(self.data)
        initial_train_end = int(total_games * self.initial_train_size)

        self.initial_train_data = self.data.iloc[:initial_train_end].copy()
        self.backtest_data = self.data.iloc[initial_train_end:].copy()

        # Separate features and target for the initial training
        self.X_train = self.initial_train_data.drop(
            columns=[self.target_column, "Game_Date", "Game_PK", "Run_Diff", "Home_Win"],
            errors='ignore'
        )
        self.y_train = self.initial_train_data[self.target_column]

    def select_model(self):
        """Selects the base model and builds the pipeline (with KNNImputer & MissingIndicator)."""
        # Identify numeric/categorical columns from X_train
        numeric_cols = self.X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in self.moneyline_columns]

        categorical_cols = self.X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in self.moneyline_columns]

        # Define preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ("union", FeatureUnion([
                ("knn", Pipeline([
                    ("imputer", KNNImputer(n_neighbors=5)),
                ])),
                ("indicator", MissingIndicator(features="all", error_on_new=False))
            ])),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # Select the model
        if self.model_type == "logistic_regression":
            base_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif self.model_type == "xgboost":
            base_model = xgb.XGBClassifier(eval_metric='logloss', random_state=self.random_state)
        elif self.model_type == "random_forest":
            base_model = RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == "gradient_boosting":
            base_model = GradientBoostingClassifier(random_state=self.random_state)
        elif self.model_type == "svm":
            base_model = SVC(probability=True, random_state=self.random_state)
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported.")

        # Build the pipeline
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", base_model)
        ])

    def train_model(self):
        """Fit the pipeline on the training data (which ensures the KNNImputer is fit only on X_train)."""
        self.pipeline.fit(self.X_train, self.y_train)
        # Now that the pipeline is fit, we can extract feature names if needed:
        self.feature_names = self.pipeline['preprocessor'].get_feature_names_out()

    def tune_hyperparameters(self):
        """Example of hyperparameter tuning that also avoids data leakage."""
        # Define parameter grids for different model types
        if self.model_type == "logistic_regression":
            param_grid = {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs"]
            }
        elif self.model_type == "xgboost":
            param_grid = {
                "classifier__n_estimators": [50],
                "classifier__max_depth": [3],
                "classifier__learning_rate": [0.05]
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
            # If no grid is defined for your custom model, skip tuning
            return

        print("Starting hyperparameter tuning on initial training data...")

        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=5,              # reduce if data is small
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        self.best_params = grid_search.best_params_
        print("Best hyperparameters found:", self.best_params)

        # Update pipeline with the best estimator
        self.pipeline = grid_search.best_estimator_

        # Optionally calibrate the best classifier
        best_classifier = self.pipeline.named_steps['classifier']
        calibrated_model = CalibratedClassifierCV(
            estimator=best_classifier,
            method='sigmoid',
            cv=5,
            n_jobs=-1
        )

        # Rebuild pipeline with calibration
        self.pipeline = Pipeline([
            ("preprocessor", self.pipeline.named_steps['preprocessor']),
            ("calibrated_classifier", calibrated_model)
        ])

        # Final fit on the training set
        self.pipeline.fit(self.X_train, self.y_train)

        print("Hyperparameter tuning and calibration complete.\n")

    def update_model(self, X_new, y_new):
        """Optionally update the model daily with new data (if self.update_model_daily is True)."""
        self.X_train = pd.concat([self.X_train, X_new], ignore_index=True)
        self.y_train = pd.concat([self.y_train, y_new], ignore_index=True)

        self.pipeline.fit(self.X_train, self.y_train)

    def predict_proba_single_game(self, game_features):
        """Convenience method for a single game."""
        proba = self.pipeline.predict_proba(game_features)[0][1]  # Probability of home win
        return proba

    def adjust_bookmaker_probabilities(self, home_odds, away_odds):
        """Converts moneyline odds to implied probabilities for home & away, then normalizes."""
        decimal_home_odds = self.moneyline_to_decimal(home_odds)
        decimal_away_odds = self.moneyline_to_decimal(away_odds)

        # Calculate implied probabilities
        implied_prob_home = 1 / decimal_home_odds
        implied_prob_away = 1 / decimal_away_odds

        # Sum of implied probabilities (overround)
        sum_implied_probs = implied_prob_home + implied_prob_away

        # Adjust probabilities so they sum to 1
        adjusted_prob_home = implied_prob_home / sum_implied_probs
        adjusted_prob_away = implied_prob_away / sum_implied_probs

        return adjusted_prob_home, adjusted_prob_away

    def run_backtest(self, initial_bankroll: float = 10000.0):
        """Backtest that places bets on **all** games each day."""
        print("Training the initial model...")
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
            # Extract features for all games on this date
            game_features_df = group[feature_columns]
            home_odds = group[self.moneyline_columns[0]].values  # e.g., 'home_odds'
            away_odds = group[self.moneyline_columns[1]].values  # e.g., 'away_odds'
            actual_outcomes = group[self.target_column].values   # 1 = home win, 0 = away win
            game_ids = group['Game_PK'].values

            # Batch predict probabilities of home win
            prob_home_win = self.pipeline.predict_proba(game_features_df)[:, 1]

            # Adjust bookmaker probabilities
            adjusted_prob_home, adjusted_prob_away = self.adjust_bookmaker_probabilities(home_odds, away_odds)

            # Collect for global evaluation
            all_model_probs.extend(prob_home_win)
            all_actual_outcomes.extend(actual_outcomes)
            all_bookmaker_probs.extend(adjusted_prob_home)  # or away, your choice

            # --- NEW: Bet on **every** game in the group ---
            for i in range(len(group)):
                game_id = game_ids[i]
                p_home = prob_home_win[i]
                p_book_home = adjusted_prob_home[i]
                actual_outcome = actual_outcomes[i]
                ml_home = home_odds[i]
                ml_away = away_odds[i]

                # Decide on which team to bet based on higher predicted probability
                if p_home > 0.5:
                    bet_on = 'home'
                    prob = p_home
                    moneyline = ml_home
                    predicted_label = 1  # Home
                else:
                    bet_on = 'away'
                    prob = 1 - p_home
                    moneyline = ml_away
                    predicted_label = 0  # Away

                # Convert moneyline to decimal odds
                decimal_odds = self.moneyline_to_decimal(moneyline)

                # Calculate the Kelly fraction for that bet
                f_opt = self.kelly_criterion(prob=prob, odds=decimal_odds)
                bet_amount = self.kelly_fraction * f_opt * bankroll if f_opt > 0 else 0

                # Potential payout
                potential_payout = bet_amount * (decimal_odds - 1)

                # Did we win?
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

                backtest_profits.append(profit)
                backtest_predictions.append(predicted_label)
                backtest_actuals.append(actual_outcome)

                # Record the result for this game
                results.append({
                    'game_id': game_id,
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

            # Update model daily if desired, using *all* games from current date
            if self.update_model_daily:
                X_new = game_features_df
                y_new = pd.Series(actual_outcomes)
                self.update_model(X_new=X_new, y_new=y_new)

        # Store probabilities and outcomes for overall evaluation
        self.all_model_probs = all_model_probs
        self.all_actual_outcomes = all_actual_outcomes
        self.all_bookmaker_probs = all_bookmaker_probs

        # Convert results to a DataFrame
        backtest_results = pd.DataFrame(results)

        # Keep these for reporting/evaluation
        self.backtest_predictions = backtest_predictions
        self.backtest_actuals = backtest_actuals
        self.backtest_profits = backtest_profits

        return backtest_results

    @staticmethod
    def kelly_criterion(prob: float, odds: float) -> float:
        """Kelly fraction for a given probability `prob` and decimal odds `odds`."""
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
        """
        Evaluate the backtest results (profit, ROI, etc.) and compare
        model vs. bookmaker Brier scores, log loss, AUC, etc.
        """
        total_profit = backtest_results['profit'].sum()
        final_bankroll = backtest_results['bankroll'].iloc[-1] if not backtest_results.empty else initial_bankroll
        roi = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
        total_bets = len(backtest_results)
        total_wins = backtest_results['bet_won'].sum()
        win_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0

        # Perform statistical tests on profits
        profits = np.array(self.backtest_profits)
        profits = profits[profits != 0]  # Remove zero-profits bets if needed

        # T-test
        t_stat, t_p_value = ttest_1samp(profits, popmean=0, alternative='greater')

        # Wilcoxon test
        try:
            w_stat, w_p_value = wilcoxon(profits - 0, alternative='greater')
        except ValueError:
            w_stat, w_p_value = np.nan, np.nan

        # Mann-Whitney U-test
        try:
            zero_profits = np.zeros_like(profits)
            u_stat, u_p_value = mannwhitneyu(profits, zero_profits, alternative='greater')
        except ValueError:
            u_stat, u_p_value = np.nan, np.nan

        # Model vs. bookmaker metrics
        model_brier = brier_score_loss(self.all_actual_outcomes, self.all_model_probs)
        bookmaker_brier = brier_score_loss(self.all_actual_outcomes, self.all_bookmaker_probs)

        model_log_loss = log_loss(self.all_actual_outcomes, self.all_model_probs)
        bookmaker_log_loss = log_loss(self.all_actual_outcomes, self.all_bookmaker_probs)

        model_auc = roc_auc_score(self.all_actual_outcomes, self.all_model_probs)
        bookmaker_auc = roc_auc_score(self.all_actual_outcomes, self.all_bookmaker_probs)

        # Diebold-Mariano style test on Brier Score differences
        loss_diff = np.array([
            (mp - ao) ** 2 - (bp - ao) ** 2
            for mp, bp, ao in zip(self.all_model_probs, self.all_bookmaker_probs, self.all_actual_outcomes)
        ])
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
        """Simple Diebold-Mariano statistic for difference in loss."""
        T = len(loss_diff)
        mean_ld = np.mean(loss_diff)
        var_ld = np.var(loss_diff, ddof=1)
        dm_stat = mean_ld / np.sqrt(var_ld / T)
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
        # Convert probabilities to class labels (threshold=0.5)
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

        classification_rep = classification_report(self.all_actual_outcomes, all_predicted_labels, zero_division=0)
        conf_matrix = confusion_matrix(self.all_actual_outcomes, all_predicted_labels)

        feature_importances = self.get_feature_importances()

        results = {
            'Accuracy_Metrics': accuracy_metrics,
            'Classification_Report': classification_rep,
            'Confusion_Matrix': conf_matrix,
            'Backtest_Evaluation': backtest_evaluation,
            'Backtest_Results': backtest_results,
            'Feature_Importances': feature_importances
        }

        self.plot_calibration_curve()
        self.generate_report(results)
        self.plot_bankroll_over_time(backtest_results)

        return results

    def get_feature_importances(self):
        """Extract (and optionally print) feature importances if available."""
        if "calibrated_classifier" not in self.pipeline.named_steps:
            return None

        calibrated_classifier = self.pipeline.named_steps['calibrated_classifier']
        importances_list = []
        feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()

        if self.model_type in ["random_forest", "gradient_boosting", "xgboost"]:
            for calibrated_clf in calibrated_classifier.calibrated_classifiers_:
                estimator = calibrated_clf.estimator
                if hasattr(estimator, 'feature_importances_'):
                    importances_list.append(estimator.feature_importances_)
                else:
                    print("Estimator does not have feature_importances_ attribute.")
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
        """Generate a text report summarizing the backtest."""
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

            kelly_info = (
                f"Kelly fraction used: {self.kelly_fraction} (Full Kelly)\n"
                if self.kelly_fraction == 1.0
                else f"Kelly fraction used: {self.kelly_fraction} (Fractional Kelly)\n"
            )
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
        """Plot bankroll over time with a log-scaled y-axis."""
        if backtest_results.empty:
            print("No results to plot.")
            return

        try:
            dates = pd.to_datetime(backtest_results['date'])
            bankroll = backtest_results['bankroll']

            plt.figure(figsize=(12, 6))
            plt.plot(dates, bankroll, label='Bankroll', marker='o', markersize=4, linewidth=2)
            plt.yscale('log')  # Log scale for y-axis
            plt.title('Bankroll Over Time (Log-Scaled)')
            plt.xlabel('Date')
            plt.ylabel('Bankroll (Log Scale)')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plot_file = os.path.join(self.output_folder, f'bankroll_over_time_{timestamp}.png')
            plt.savefig(plot_file)
            plt.close()

            print(f"Bankroll over time plot saved to {plot_file}")
        except Exception as e:
            print(f"Error plotting bankroll over time: {e}")

    def get_trained_pipeline(self):
        """Return the final trained (calibrated) pipeline."""
        return self.pipeline

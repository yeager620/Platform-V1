import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Sklearn & SciKeras Imports
from sklearn.impute import KNNImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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

# Keras + SciKeras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier


def create_keras_model(
        n_features: int,
        hidden_layer_sizes=(256, 128),
        dropout_rate=0.2,
        learning_rate=1e-4,
) -> Sequential:
    """
    Builds a Keras Sequential model with Batch Normalization + Dropout.
    """
    model = Sequential()

    # First hidden layer
    model.add(Dense(hidden_layer_sizes[0], activation='relu', input_shape=(n_features,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Additional hidden layers
    for layer_size in hidden_layer_sizes[1:]:
        model.add(Dense(layer_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))  # Probability of the positive class

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


class BacktestingEngine:
    def __init__(
            self,
            data: pd.DataFrame,
            target_column: str,
            moneyline_columns: list,
            initial_train_size: float = 0.2,
            kelly_fraction: float = 1.0,  # Full Kelly by default
            update_model: bool = False,
            output_folder: str = "./backtest_reports",
            random_state: int = 42
    ):
        """
        A simplified backtesting engine that only uses a neural network (Keras + SciKeras)
        with dropout & batch normalization. Performs calibration to get well-calibrated
        probabilities and minimal log loss.
        """
        self.data = data.copy()

        self.target_column = target_column
        self.run_diff_col = "Run_Diff"
        self.moneyline_columns = moneyline_columns

        self.initial_train_size = initial_train_size
        self.kelly_fraction = kelly_fraction
        self.output_folder = output_folder
        self.random_state = random_state
        self.update_model_daily = update_model

        # Initialize placeholders
        self.pipeline = None
        self.initial_train_data = None
        self.backtest_data = None
        self.feature_names = None  # For introspection if needed
        self.best_params = None

        # Placeholders for probabilities and actual outcomes
        self.all_model_probs = []
        self.all_actual_outcomes = []
        self.all_bookmaker_probs = []

        # Basic data prep
        self._prepare_data()
        self.split_data()
        self.select_model()

    def _prepare_data(self):
        """Sort data chronologically and drop rows with missing moneylines."""
        if 'Game_Date' in self.data.columns:
            self.data.sort_values(by='Game_Date', inplace=True)
        else:
            raise ValueError("Data must contain a 'Game_Date' column for chronological sorting.")

        # Example: treat 'park_id' as categorical
        if 'park_id' in self.data.columns:
            self.data['park_id'] = self.data['park_id'].astype(str)

        # Drop rows with missing moneyline values
        self.data.dropna(subset=self.moneyline_columns, inplace=True)

    def split_data(self):
        """Chronological split into initial training and backtest set."""
        total_games = len(self.data)
        initial_train_end = int(total_games * self.initial_train_size)

        self.initial_train_data = self.data.iloc[:initial_train_end].copy()
        self.backtest_data = self.data.iloc[initial_train_end:].copy()

        self.X_train = self.initial_train_data.drop(
            columns=[self.target_column, "Game_Date", "Game_PK", "Run_Diff", "Home_Win"],
            errors='ignore'
        )
        self.y_train = self.initial_train_data[self.target_column]

    def select_model(self):
        """Build a Pipeline with preprocessing + Keras neural network."""
        # Identify numeric/categorical columns
        numeric_cols = self.X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in self.moneyline_columns]

        categorical_cols = self.X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in self.moneyline_columns]

        # Preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ("union", FeatureUnion([
                ("knn", Pipeline([
                    ("imputer", KNNImputer(n_neighbors=5)),
                ])),
                ("indicator", MissingIndicator(features="missing-only"))
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

        # Keras model (via SciKeras)
        base_model = KerasClassifier(
            model=create_keras_model,
            hidden_layer_sizes=(256, 128),
            dropout_rate=0.2,
            learning_rate=1e-4,
            epochs=50,
            batch_size=32,
            verbose=0,
            random_state=self.random_state
        )

        # Build the pipeline
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", base_model)
        ])

    def tune_hyperparameters(self):
        """
        Example hyperparameter tuning for the Keras neural network + final calibration.
        """
        # Parameter grid for the neural network
        param_grid = {
            "classifier__model__hidden_layer_sizes": [
                (256, 128),
                (128, 64),
                (64, 32, 16)
            ],
            "classifier__model__dropout_rate": [0.2, 0.3],
            "classifier__model__learning_rate": [1e-4, 1e-3],
            "classifier__epochs": [50, 100],
            "classifier__batch_size": [32, 64],
        }

        print("Starting hyperparameter tuning on initial training data...")

        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=5,  # n-fold CV
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        self.best_params = grid_search.best_params_
        print("Best hyperparameters found:", self.best_params)

        # Update pipeline with the best estimator
        self.pipeline = grid_search.best_estimator_

        # Calibrate the best classifier
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

        # Final fit on the entire training set
        self.pipeline.fit(self.X_train, self.y_train)
        print("Hyperparameter tuning and calibration complete.\n")

    def update_model(self, X_new, y_new):
        """
        Optionally update (retrain) the model daily with the new data.
        This re-fits the calibrated pipeline on the combined dataset.
        """
        self.X_train = pd.concat([self.X_train, X_new], ignore_index=True)
        self.y_train = pd.concat([self.y_train, y_new], ignore_index=True)
        self.pipeline.fit(self.X_train, self.y_train)

    def predict_proba_single_game(self, game_features):
        """Predict the probability of a home win for a single game."""
        # Note: 'calibrated_classifier' expects columns in the same order as training
        proba = self.pipeline.predict_proba(game_features)[0][1]
        return proba

    def adjust_bookmaker_probabilities(self, home_odds, away_odds):
        """Convert moneylines to implied probabilities & normalize to remove overround."""
        decimal_home_odds = self.moneyline_to_decimal(home_odds)
        decimal_away_odds = self.moneyline_to_decimal(away_odds)

        implied_prob_home = 1 / decimal_home_odds
        implied_prob_away = 1 / decimal_away_odds

        sum_implied_probs = implied_prob_home + implied_prob_away
        adjusted_prob_home = implied_prob_home / sum_implied_probs
        adjusted_prob_away = implied_prob_away / sum_implied_probs

        return adjusted_prob_home, adjusted_prob_away

    def run_backtest(self, initial_bankroll: float = 10000.0):
        """
        Run the backtest by training once on the initial set, then picking one game daily.
        Optionally update the model each day if self.update_model_daily is True.
        """
        print("Training the initial model with hyperparameter tuning + calibration...")
        self.tune_hyperparameters()
        print("Initial training completed.\n")

        bankroll = initial_bankroll
        results = []

        backtest_predictions = []
        backtest_actuals = []
        backtest_profits = []

        all_model_probs = []
        all_actual_outcomes = []
        all_bookmaker_probs = []

        # Exclude certain columns (moneylines, date, target, etc.) from the features
        excluded_columns = self.moneyline_columns + [self.target_column, 'Game_Date', 'Game_PK']
        feature_columns = [col for col in self.backtest_data.columns if col not in excluded_columns]

        # Group games by date to simulate daily bets
        date_column = 'Game_Date' if 'Game_Date' in self.backtest_data.columns else 'date'
        grouped = self.backtest_data.groupby(date_column)

        # Iterate day-by-day
        for date, group in tqdm(grouped, total=grouped.ngroups, desc="Processing Dates"):
            game_features_df = group[feature_columns]
            home_odds = group[self.moneyline_columns[0]].values
            away_odds = group[self.moneyline_columns[1]].values
            actual_outcomes = group[self.target_column].values  # 1=home, 0=away
            game_ids = group['Game_PK'].values

            # Model probabilities
            prob_home_win = self.pipeline.predict_proba(game_features_df)[:, 1]
            # Bookmaker implied probabilities
            adjusted_prob_home, _ = self.adjust_bookmaker_probabilities(home_odds, away_odds)

            # Record for global metrics
            all_model_probs.extend(prob_home_win)
            all_actual_outcomes.extend(actual_outcomes)
            all_bookmaker_probs.extend(adjusted_prob_home)

            # Select one "best" game to bet on (largest difference from bookmaker)
            prob_diff = np.abs(prob_home_win - adjusted_prob_home)
            best_game_idx = np.argmax(prob_diff)

            best_game_id = game_ids[best_game_idx]
            best_game_prob = prob_home_win[best_game_idx]
            best_game_actual_outcome = actual_outcomes[best_game_idx]
            best_game_row = group.iloc[best_game_idx]

            if best_game_prob > 0.5:
                bet_on = 'home'
                prob = best_game_prob
                moneyline = best_game_row[self.moneyline_columns[0]]
                predicted_label = 1
            else:
                bet_on = 'away'
                prob = 1 - best_game_prob
                moneyline = best_game_row[self.moneyline_columns[1]]
                predicted_label = 0

            backtest_predictions.append(predicted_label)
            backtest_actuals.append(best_game_actual_outcome)

            # Kelly stake
            decimal_odds = self.moneyline_to_decimal(moneyline)
            f_opt = self.kelly_criterion(prob, decimal_odds)
            bet_amount = self.kelly_fraction * f_opt * bankroll if f_opt > 0 else 0

            potential_payout = bet_amount * (decimal_odds - 1)
            if bet_on == 'home':
                bet_won = 1 if best_game_actual_outcome == 1 else 0
            else:
                bet_won = 1 if best_game_actual_outcome == 0 else 0

            if bet_won:
                bankroll += potential_payout
                profit = potential_payout
            else:
                bankroll -= bet_amount
                profit = -bet_amount

            backtest_profits.append(profit)

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

            # Optionally update the model with today's games
            if self.update_model_daily:
                X_new = game_features_df
                y_new = pd.Series(actual_outcomes)
                self.update_model(X_new, y_new)

        # Store for global evaluation
        self.all_model_probs = all_model_probs
        self.all_actual_outcomes = all_actual_outcomes
        self.all_bookmaker_probs = all_bookmaker_probs
        self.backtest_predictions = backtest_predictions
        self.backtest_actuals = backtest_actuals
        self.backtest_profits = backtest_profits

        backtest_results = pd.DataFrame(results)
        return backtest_results

    @staticmethod
    def kelly_criterion(prob: float, odds: float) -> float:
        """Kelly fraction for a given probability 'prob' and decimal odds 'odds'."""
        if odds <= 1:
            return 0.0
        return max((prob * (odds - 1) - (1 - prob)) / (odds - 1), 0.0)

    @staticmethod
    def moneyline_to_decimal(moneyline):
        """Convert moneyline odds to decimal odds."""
        moneyline = np.asarray(moneyline)
        decimal_odds = np.where(
            moneyline > 0,
            (moneyline / 100) + 1,
            np.where(
                moneyline < 0,
                (100 / np.abs(moneyline)) + 1,
                1.0  # Fallback for edge cases
            )
        )
        return decimal_odds

    def evaluate_backtest(self, backtest_results: pd.DataFrame, initial_bankroll: float):
        """Evaluate performance metrics (ROI, Brier Score, Log Loss, etc.)."""
        if backtest_results.empty:
            print("No bets were placed, backtest results are empty.")
            return {}

        total_profit = backtest_results['profit'].sum()
        final_bankroll = backtest_results['bankroll'].iloc[-1]
        roi = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
        total_bets = len(backtest_results)
        total_wins = backtest_results['bet_won'].sum()
        win_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0

        profits = np.array(self.backtest_profits)
        profits = profits[profits != 0]  # remove zero-profit bets if desired

        # Basic significance tests
        t_stat, t_p_value = ttest_1samp(profits, popmean=0, alternative='greater')
        try:
            w_stat, w_p_value = wilcoxon(profits - 0, alternative='greater')
        except ValueError:
            w_stat, w_p_value = np.nan, np.nan

        try:
            zero_profits = np.zeros_like(profits)
            u_stat, u_p_value = mannwhitneyu(profits, zero_profits, alternative='greater')
        except ValueError:
            u_stat, u_p_value = np.nan, np.nan

        # Probability-based metrics
        model_brier = brier_score_loss(self.all_actual_outcomes, self.all_model_probs)
        bookmaker_brier = brier_score_loss(self.all_actual_outcomes, self.all_bookmaker_probs)

        model_log_loss = log_loss(self.all_actual_outcomes, self.all_model_probs)
        bookmaker_log_loss = log_loss(self.all_actual_outcomes, self.all_bookmaker_probs)

        model_auc = roc_auc_score(self.all_actual_outcomes, self.all_model_probs)
        bookmaker_auc = roc_auc_score(self.all_actual_outcomes, self.all_bookmaker_probs)

        # Diebold-Mariano on difference in Brier scores
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
        """Simple Diebold-Mariano statistic for difference in forecast errors."""
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
        """Plot calibration curves for model vs. bookmaker."""
        plt.figure(figsize=(10, 5))

        prob_pred_model, prob_true_model = calibration_curve(self.all_actual_outcomes, self.all_model_probs, n_bins=10)
        plt.plot(prob_pred_model, prob_true_model, marker='o', label='Model')

        prob_pred_book, prob_true_book = calibration_curve(self.all_actual_outcomes, self.all_bookmaker_probs,
                                                           n_bins=10)
        plt.plot(prob_pred_book, prob_true_book, marker='s', label='Bookmaker')

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
        """Run backtest -> evaluate -> generate report -> plot curves."""
        print("Starting backtest simulation...")
        backtest_results = self.run_backtest(initial_bankroll=initial_bankroll)
        print("Backtest simulation completed.\n")

        print("Evaluating backtest performance...")
        backtest_evaluation = self.evaluate_backtest(backtest_results, initial_bankroll=initial_bankroll)
        print("Backtest evaluation completed.\n")

        print("Evaluating model accuracy on all predictions...")
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

        results = {
            'Accuracy_Metrics': accuracy_metrics,
            'Classification_Report': classification_rep,
            'Confusion_Matrix': conf_matrix,
            'Backtest_Evaluation': backtest_evaluation,
            'Backtest_Results': backtest_results
        }

        self.plot_calibration_curve()
        self.generate_report(results)
        self.plot_bankroll_over_time(backtest_results)

        return results

    def generate_report(self, results):
        """Generate a text report summarizing backtest metrics & final results."""
        print("Generating test report...")

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        kelly_str = f"kelly_{self.kelly_fraction}"
        report_file_name = f"backtest_report_neural_{kelly_str}_{timestamp}.txt"
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

            if self.kelly_fraction == 1.0:
                f.write("Kelly fraction used: 1.0 (Full Kelly)\n")
            else:
                f.write(f"Kelly fraction used: {self.kelly_fraction} (Fractional Kelly)\n")

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

            f.write("\n\nAdditional Information:\n")
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
            plt.yscale('log')  # Log scale for the bankroll
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
        """Return the final trained (possibly calibrated) pipeline."""
        return self.pipeline

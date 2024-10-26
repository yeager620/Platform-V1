import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
            model_type: str = "logistic_regression",
            test_size: float = 0.2,
            random_state: int = 42,
    ):
        """
        Initializes the BacktestingEngine with the dataset and model parameters.

        Parameters:
            data (pd.DataFrame): The dataset containing feature vectors and target variable.
            target_column (str): The name of the target variable column.
            model_type (str): The type of model to use. Defaults to 'logistic_regression'.
            test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int): Controls the shuffling applied to the data before splitting. Defaults to 42.
        """
        self.data = data.copy()
        self.target_column = target_column
        self.model_type = model_type.lower()
        self.test_size = test_size
        self.random_state = random_state

        # Initialize placeholders
        self.model = None
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Preprocessing and model selection
        self.preprocess_data()
        self.split_data()
        self.select_model()

    def preprocess_data(self):
        """
        Preprocesses the data by handling missing values, encoding categorical variables,
        and scaling numerical features.
        """
        # Separate features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

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

        # Update the data with preprocessing
        self.X = X
        self.y = y

    def split_data(self):
        """
        Splits the data into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,  # To maintain the distribution of the target variable
        )

    def select_model(self):
        """
        Selects and initializes the machine learning model based on the specified model_type.
        Defaults to Logistic Regression.
        """
        if self.model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=self.random_state)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(random_state=self.random_state)
        elif self.model_type == "svm":
            self.model = SVC(probability=True, random_state=self.random_state)
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported.")

        # Create a pipeline that first preprocesses the data and then fits the model
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", self.model)
        ])

    def train_model(self):
        """
        Trains the machine learning model using the training data.
        """
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self):
        """
        Generates predictions on the test data.

        Returns:
            y_pred (np.ndarray): Predicted class labels.
            y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
        """
        y_pred = self.pipeline.predict(self.X_test)
        if hasattr(self.pipeline.named_steps["classifier"], "predict_proba"):
            y_pred_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
        else:
            # For models like SVM without predict_proba by default
            y_pred_proba = self.pipeline.decision_function(self.X_test)
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())  # Normalize
        return y_pred, y_pred_proba

    def evaluate_model(self, y_pred, y_pred_proba):
        """
        Evaluates the model's performance using various classification metrics.

        Parameters:
            y_pred (np.ndarray): Predicted class labels.
            y_pred_proba (np.ndarray): Predicted probabilities for the positive class.

        Returns:
            metrics_dict (dict): Dictionary containing evaluation metrics.
            classification_rep (str): Detailed classification report.
            conf_matrix (np.ndarray): Confusion matrix.
        """
        metrics_dict = {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision": precision_score(self.y_test, y_pred, zero_division=0),
            "Recall": recall_score(self.y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(self.y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(self.y_test, y_pred_proba),
        }

        classification_rep = classification_report(self.y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        return metrics_dict, classification_rep, conf_matrix

    def run_backtest(self):
        """
        Executes the full backtesting pipeline: training, prediction, and evaluation.

        Returns:
            results (dict): Dictionary containing evaluation metrics, classification report, and confusion matrix.
        """
        print("Training the model...")
        self.train_model()
        print("Model training completed.\n")

        print("Making predictions on the test set...")
        y_pred, y_pred_proba = self.predict()
        print("Predictions completed.\n")

        print("Evaluating the model...")
        metrics, report, confusion = self.evaluate_model(y_pred, y_pred_proba)
        print("Evaluation completed.\n")

        results = {
            "Metrics": metrics,
            "Classification_Report": report,
            "Confusion_Matrix": confusion,
        }

        return results

    def get_trained_pipeline(self):
        """
        Returns the trained pipeline for further use or inspection.

        Returns:
            pipeline (Pipeline): Trained scikit-learn Pipeline object.
        """
        return self.pipeline

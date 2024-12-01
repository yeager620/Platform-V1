import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class MaqLogisticRegression:
    def __init__(self, data: pd.DataFrame, target_column: str, features_columns: list, random_state: int = 42):
        """
        Initializes the CustomLogisticRegression with the dataset, target variable, and feature columns.

        Parameters:
            data (pd.DataFrame): The dataset containing features and the target variable.
            target_column (str): The name of the target variable column.
            features_columns (list): List of column names to be used as features.
            random_state (int): Controls the randomness of the logistic regression model.
        """
        self.data = data
        self.target_column = target_column
        self.features_columns = features_columns
        self.random_state = random_state
        self.model = None
        self.pipeline = None
        self.preprocessor = None

        self._prepare_data()

    def _prepare_data(self):
        """
        Prepares the data for training by handling missing values, encoding categorical variables, and scaling numerical features.
        """
        X = self.data[self.features_columns]
        y = self.data[self.target_column]

        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Define preprocessing steps
        numerical_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Combine transformers using ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # Prepare feature and target arrays for model training
        self.X = X
        self.y = y

    def train_model(self):
        """
        Trains the logistic regression model using the prepared data.
        """
        # Initialize the Logistic Regression model
        base_model = LogisticRegression(random_state=self.random_state)

        # Wrap the base model with CalibratedClassifierCV for probability calibration
        calibrated_model = CalibratedClassifierCV(base_estimator=base_model, method='sigmoid', cv=5)

        # Create a pipeline that first preprocesses the data and then fits the calibrated model
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("calibrated_classifier", calibrated_model)
        ])

        # Fit the model
        self.pipeline.fit(self.X, self.y)

    def predict_proba(self, X_new):
        """
        Predicts probabilities for the target classes.

        Parameters:
            X_new (pd.DataFrame): New data for which to predict probabilities.

        Returns:
            np.array: Array of probabilities for the target classes.
        """
        return self.pipeline.predict_proba(X_new)

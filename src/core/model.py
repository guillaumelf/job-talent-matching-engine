"""File containing the class used to train the model used in the project."""

import logging
import typing

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src import constants


class MatchingModel:
    """A class implementing the machine learning model used to predict the label."""

    def __init__(self, dataset: typing.Optional[pd.DataFrame] = None) -> None:
        """Initialize the model with the dataset."""
        if dataset:
            self.dataset = dataset
        # If the model doesn't exist, train it
        if not self.check_model_exists:
            if not self.dataset:
                raise ValueError("A dataset must be provided to train the model.")
            logging.warning("Model not found. Training a new model.")
            self.train_model()
        # Load the model
        self.model = joblib.load(constants.MODEL_FILE_PATH)

    def check_label_is_in_columns(self) -> None:
        """Check if the 'label' column is present in the dataset."""
        if "label" not in self.dataset.columns:
            raise ValueError("The 'label' column is missing from the dataset.")

    def check_model_exists(self) -> bool:
        """Check if the model file exists."""
        return constants.MODEL_FILE_PATH.exists()

    def get_train_test_sets(self) -> tuple:
        """Split the dataset into train and test sets."""
        self.check_label_is_in_columns()
        x = self.dataset.drop(columns=["label"])
        y = self.dataset["label"]
        return train_test_split(x, y, test_size=constants.TEST_DATASET_SIZE)

    def train_model(self) -> None:
        """Train the machine learning model and save it to disk."""
        # Initialize the model
        model = RandomForestClassifier(
            n_estimators=constants.MODEL_ITERATIONS_NUMBER, random_state=42
        )

        # Split the dataset
        x_train, x_test, y_train, y_test = self.get_train_test_sets()

        # Train the model
        model.fit(x_train, y_train)

        # Evaluate the model's performance
        score = model.score(x_test, y_test)
        logging.info("Accuracy score : %.4f", score)

        # Save the model
        joblib.dump(model, constants.MODEL_FILE_PATH)

    def predict_label(self, data: pd.DataFrame | np.ndarray) -> bool:
        """Predict the label for a given data point."""
        return self.model.predict(data)

    def predict_score(self, data: pd.DataFrame | np.ndarray) -> float:
        """Predict the label for a given data point."""
        return self.model.predict_proba(data)

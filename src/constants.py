"""File containing all the constants used in the project."""

import pathlib

# Paths to the data folder and data file
DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / "data"
DATA_FILE_PATH: pathlib.Path = DATA_FOLDER_PATH / "data.json"

# Encoding values for each degree, used during the feature engineering phase
DEGREE_ENCODING: dict[str, int] = {
    "none": 0,
    "apprenticeship": 1,
    "bachelor": 2,
    "master": 3,
    "doctorate": 4,
}

# Model's number of iterations
MODEL_ITERATIONS_NUMBER: int = 100

# Model's file path
MODEL_FILE_PATH: pathlib.Path = DATA_FOLDER_PATH / "model.joblib"

# Test dataset relative size
TEST_DATASET_SIZE: float = 0.2

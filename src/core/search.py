"""A file containing the Search class implementing the matching mechanism."""

import itertools
import multiprocessing as mp

from src.core import model
from src.utils import features_extraction


class Search:
    """A class implementing the search engine used to match talents with jobs."""

    def __init__(self) -> None:
        self.model_instance = model.MatchingModel()

    def match(self, talent: dict, job: dict) -> dict:
        """Match a talent with a job and return the predicted label and score.

        This method takes a talent and job as input and uses the machine learning
        model to predict the label. The input talent and job are provided as dictionaries.
        The returned dictionary has the following schema:

        {
          "talent": ...,
          "job": ...,
          "label": ...,
          "score": ...
        }

        Args:
            talent (dict): A dictionary representing the talent.
            job (dict): A dictionary representing the job.

        Returns:
            dict: A dictionary containing the talent, job, predicted label, and score.
        """
        # Initialize the data dictionary
        data = {"talent": talent, "job": job}
        # Extract features from the data
        input_features = (
            features_extraction.FeaturesExtractor().extract_features_as_array(data)
        )
        data["label"] = self.model_instance.predict_label(input_features)
        data["score"] = self.model_instance.model.predict_proba(input_features)[0][1]
        return data

    def match_bulk(self, talents: list[dict], jobs: list[dict]) -> list[dict]:
        """Match multiple talents with multiple jobs.

        This method takes a list of talents and a list of jobs as input and uses a machine
        learning model to predict the label for each combination. The method returns a list
        of dictionaries, sorted in descending order by score. Each dictionary in the list
        represents a matched talent and job pair, and has the following schema:

        {
            "talent": ...,
            "job": ...,
            "label": ...,
            "score": ...
        }

        Args:
            talents (list[dict]): A list of dictionaries representing talents.
            jobs (list[dict]): A list of dictionaries representing jobs.

        Returns:
            list[dict]: A list of dictionaries representing matched talent and job pairs.
        """
        # We use multiprocessing to speed up the computation
        with mp.Pool(mp.cpu_count() - 1) as pool:
            return sorted(
                pool.starmap(self.match, itertools.product(talents, jobs)),
                key=lambda x: x["score"],
                reverse=True,
            )

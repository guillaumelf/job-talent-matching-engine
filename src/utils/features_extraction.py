"""File containing util functions used to extract features from the data."""

import numpy as np
import pandas as pd

from src import constants


class FeaturesExtractor:
    """A class implementening the feature extraction functions.

    These functions give us 8 input features for our ML model :
    - must_have_languages_overlap
    - optional_languages_spoken
    - job_roles_overlap
    - job_roles_number
    - talent_has_seniority_requirement
    - talent_has_min_degree_requirement
    - salary_expectation_matches_offer
    - salary_expectation_gap

    The details for each feature can be found in the docstrings of the corresponding function.
    """

    def __init__(self) -> None:
        """Initialize the class with arguments if any"""

    def get_dataset(self, batch: dict) -> pd.DataFrame:
        """Extract features from raw data and return a Dataframe."""
        return pd.DataFrame.from_records(
            [self.extract_features_as_dict(data) for data in batch]
        )

    def extract_features_as_dict(self, data: dict) -> dict:
        """Extract features from raw data, contained in a dict."""
        return {
            "must_have_languages_overlap": self.compute_must_have_languages_overlap(
                data
            ),
            "optional_languages_spoken": self.optional_languages_spoken(data),
            "job_roles_overlap": self.compute_job_roles_overlap(data),
            "job_roles_number": self.compute_job_roles_number(data),
            "talent_has_seniority_requirement": self.talent_has_seniority_requirement(
                data
            ),
            "talent_has_min_degree_requirement": self.talent_has_min_degree_requirement(
                data
            ),
            "salary_expectation_matches_offer": self.salary_expectation_matches_offer(
                data
            ),
            "salary_expectation_gap": self.compute_salary_expectation_gap(data),
            "label": data["label"],
        }

    def extract_features_as_array(self, data: dict) -> np.ndarray:
        """Extract features from raw data, contained in an array."""
        return np.array(
            [
                self.compute_must_have_languages_overlap(data),
                self.optional_languages_spoken(data),
                self.compute_job_roles_overlap(data),
                self.compute_job_roles_number(data),
                self.talent_has_seniority_requirement(data),
                self.talent_has_min_degree_requirement(data),
                self.salary_expectation_matches_offer(data),
                self.compute_salary_expectation_gap(data),
            ]
        ).reshape(1, -1)

    def compute_must_have_languages_overlap(self, data: dict) -> float:
        """
        Compute the overlap between the 'must have' languages of a job and a talent.

        Args:
            data (dict): A dictionary containing the talent and job information.

        Returns:
            float: The overlap between the must have languages spoken by the talent
                and the job's requirements, represented as a float between 0 and 1.
        """
        talent_languages = data["talent"]["languages"]
        job_languages = data["job"]["languages"]

        must_have_languages_matches = 0
        job_language_required = 0
        for job_language in job_languages:
            # Continue the loop if the language is not a 'must have' one
            if job_language["must_have"]:
                # Otherwise increment the counter used as a denominator
                job_language_required += 1
                for talent_language in talent_languages:
                    # If the talent speaks the language and has a good enough rating : add a match
                    if (talent_language["title"] == job_language["title"]) and (
                        talent_language["rating"] >= job_language["rating"]
                    ):
                        must_have_languages_matches += 1
        # Avoid divided by zero which would cause an error
        if job_language_required == 0:
            return 0
        return must_have_languages_matches / job_language_required

    def optional_languages_spoken(self, data: dict) -> int:
        """
        Compute the number of 'optional' languages a talent speaks compared
        to the job's requirements.

        Args:
            data (dict): A dictionary containing the talent and job information.

        Returns:
            int: The number of optional languages spoken by the talent compared
                to the job's requirements.
        """
        talent_languages_titles = [
            talent_language["title"] for talent_language in data["talent"]["languages"]
        ]
        must_have_job_languages_titles = [
            job_language["title"]
            for job_language in data["job"]["languages"]
            if job_language["must_have"]
        ]

        return len(set(talent_languages_titles) - set(must_have_job_languages_titles))

    def compute_job_roles_overlap(self, data: dict) -> float:
        """
        Compute the overlap between the 'job roles' of a job requirements and a talent.

        Args:
            data (dict): A dictionary containing the talent and job information.

        Returns:
            float: The overlap between the job roles of the talent and the job,
                represented as a float between 0 and 1.
        """
        talent_job_roles = data["talent"]["job_roles"]
        job_job_roles = data["job"]["job_roles"]

        return len(set(talent_job_roles) & set(job_job_roles)) / len(set(job_job_roles))

    def compute_job_roles_number(self, data: dict) -> int:
        """
        Compute the number of applicable job roles.

        Args:
            data (dict): A dictionary containing the talent and job information.

        Returns:
            int: The number of applicable job roles.
        """
        return len(set(data["job"]["job_roles"]))

    def talent_has_seniority_requirement(self, data: dict) -> bool:
        """
        Compute the overlap between the 'seniority' of a talent and expected ones in the job.

        Args:
            data (dict): A dictionary containing the talent and job information.

        Returns:
            bool: The overlap between the seniority of the talent and the job,
                represented as a boolean value.
        """
        talent_seniority = data["talent"]["seniority"]
        job_seniorities = data["job"]["seniorities"]

        return talent_seniority in job_seniorities

    def talent_has_min_degree_requirement(self, data: dict) -> bool:
        """
        Compute whether the talent is qualified enough for the job or not.

        Args:
            data (dict): A dictionary containing the talent and job information.

        Returns:
            bool: Whether the talent has the minimum degree required or not,
                represented as a boolean value.
        """
        return constants.DEGREE_ENCODING.get(
            data["talent"]["degree"], 0
        ) >= constants.DEGREE_ENCODING.get(data["job"]["min_degree"], 0)

    def salary_expectation_matches_offer(self, data: dict) -> bool:
        """
        Compute whether the salary expectation of the talent matches the offer of the job.

        Args:
            data (dict): A dictionary containing the talent and job information.

        Returns:
            bool: Whether the salary expectation of the talent matches the offer of the job,
                represented as a boolean value.
        """
        return data["talent"]["salary_expectation"] <= data["job"]["max_salary"]

    def compute_salary_expectation_gap(self, data: dict) -> int:
        """
        Compute the gap between the salary expectation of the talent and the offer of the job.

        Args:
            data (dict): A dictionary containing the talent and job information.

        Returns:
            int: The gap between the salary expectation of the talent and the offer of the job.
        """
        return data["job"]["max_salary"] - data["talent"]["salary_expectation"]

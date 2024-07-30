"""Main script for the job-talent-matching project."""

import json
import logging

from src import constants
from src.core import search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    logging.info("Loading data...")
    with open(constants.DATA_FILE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    talents = [sub_data["talent"] for sub_data in data][:500]
    jobs = [sub_data["job"] for sub_data in data][:500]

    # Initialize the search engine
    search = search.Search()

    # Match talents with jobs
    logging.info("Matching %s talents with %s jobs...", {len(talents)}, {len(jobs)})
    results = search.match_bulk(talents, jobs)

    logging.info("Matching completed.")
    logging.info("First 5 results:")
    logging.info(results[:5])

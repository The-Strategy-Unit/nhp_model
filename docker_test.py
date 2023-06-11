#!/opt/conda/bin/python
"""Run the model inside of the docker container"""

import argparse
import json
import logging

from model.helpers import load_params
from run_model import run_all


def progress_callback(model_id):
    """Progress callback method

    updates a file containing the status of the current model runs
    """
    current_progress = {"Inpatients": 0, "Outpatients": 0, "AaE": 0}
    filename = f"results/status-{model_id}.json"

    def save_file():
        with open(filename, "w", encoding="UTF-8") as file:
            json.dump(current_progress, file)

    save_file()

    def callback(model_type, n_completed):
        current_progress[model_type] = n_completed
        save_file()

    return callback


def main():
    """the main method"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "params_file",
        nargs="?",
        default="sample_params.json",
        help="Name of the parameters file stored in Azure",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("running model for: %s", args.params_file)

    params = load_params(f"queue/{args.params_file}")

    results_file = run_all(params, "data", progress_callback)

    logging.info("complete: %s", results_file)


def init():
    """method for calling main"""
    if __name__ == "__main__":
        main()


init()

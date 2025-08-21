"""Methods for running the NHP model in a Docker container."""

import argparse
import logging

from nhp.docker.config import Config
from nhp.docker.run import RunWithAzureStorage, RunWithLocalStorage
from nhp.model.data import Local
from nhp.model.run import run_all


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "params_file",
        nargs="?",
        default="params-sample.json",
        help="Name of the parameters file stored in Azure",
    )

    parser.add_argument(
        "--local-storage",
        "-l",
        action="store_true",
        help="Use local storage (instead of Azure)",
    )

    parser.add_argument("--save-full-model-results", action="store_true")

    return parser.parse_args()


def main(config: Config = Config()):
    """The main method."""
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.local_storage:
        runner = RunWithLocalStorage(args.params_file)
    else:
        runner = RunWithAzureStorage(args.params_file, config)

    logging.info("running model for: %s", args.params_file)
    logging.info("submitted by: %s", runner.params.get("user"))
    logging.info("model_runs:   %s", runner.params["model_runs"])
    logging.info("start_year:   %s", runner.params["start_year"])
    logging.info("end_year:     %s", runner.params["end_year"])
    logging.info("app_version:  %s", runner.params["app_version"])

    saved_files, results_file = run_all(
        runner.params,
        Local.create("data"),
        runner.progress_callback(),
        args.save_full_model_results,
    )

    runner.finish(results_file, saved_files, args.save_full_model_results)

    logging.info("complete")


def init():
    """Method for calling main."""
    if __name__ == "__main__":
        # run the model in a try catch block - ensures any exceptions that occur in the
        # multiprocessing pool are handled and logged correctly.
        # this prevents the docker container from hanging indefinitely.
        try:
            config = Config()
            main(config)
        except Exception as e:
            logging.error("An error occurred: %s", str(e))
            raise e


init()

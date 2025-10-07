"""Functions to run the model.

This module allows you to run the various models. It allows you to run a single model run of one of
the different types of models for debugging purposes, or it allows you to run all of the models in
parallel saving the results to disk.

There are existing launch profiles for vscode that use this file, or you can use it directly in the
console, e.g.

    python -m nhp.model -d data --model-run 1 -t ip

will run a single run of the inpatients model, returning the results to display.
"""

import argparse
import logging

from nhp.model.aae import AaEModel
from nhp.model.data import Local
from nhp.model.helpers import load_params
from nhp.model.inpatients import InpatientsModel
from nhp.model.outpatients import OutpatientsModel
from nhp.model.run import run_all, run_single_model_run


def _parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "params_file",
        nargs="?",
        default="queue/params-sample.json",
        help="Path to the params.json file",
    )
    parser.add_argument("-d", "--data-path", help="Path to the data", default="data")
    parser.add_argument(
        "-r", "--model-run", help="Which model iteration to run", default=1, type=int
    )
    parser.add_argument(
        "-t",
        "--type",
        default="all",
        choices=["all", "aae", "ip", "op"],
        help="Model type, either: all, ip, op, aae",
        type=str,
    )
    parser.add_argument("--save-full-model-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Main method.

    Runs when __name__ == "__main__"
    """
    # Grab the Arguments
    args = _parse_args()
    params = load_params(args.params_file)
    # define the model to run
    match args.type:
        case "all":
            logging.basicConfig(
                format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
                level=logging.INFO,
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            run_all(
                params,
                Local.create(args.data_path),
                lambda _: lambda _: None,
                args.save_full_model_results,
            )
            return
        case "aae":
            model_type = AaEModel
        case "ip":
            model_type = InpatientsModel
        case "op":
            model_type = OutpatientsModel
        case _:
            raise ValueError(f"Unknown model type: {args.type}")

    run_single_model_run(params, args.data_path, model_type, args.model_run)


def init():
    """Method for calling main."""
    if __name__ == "__main__":
        main()


init()

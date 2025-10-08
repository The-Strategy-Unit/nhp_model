"""Generate sample parameters."""

import argparse
import json
import random
from datetime import datetime

from . import load_sample_params


def _parse_args():
    parser = argparse.ArgumentParser(description="CLI for loading sample parameters.")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--scenario", required=True, help="Scenario name")
    parser.add_argument("--app-version", default="dev", help="App version (default: dev)")
    parser.add_argument("--model-runs", type=int, default=256, help="Model Runs (default: 256)")
    parser.add_argument("--start-year", type=int, default=2023, help="Start year (default: 2023)")
    parser.add_argument("--end-year", type=int, default=2041, help="End year (default: 2041)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: a random integer between 0 and 10000)",
    )

    return parser.parse_args()


def main():
    """Generate sample parameters and print them to the console."""
    args = _parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 10000)

    params = load_sample_params(
        dataset=args.dataset,
        scenario=args.scenario,
        app_version=args.app_version,
        start_year=args.start_year,
        end_year=args.end_year,
        seed=args.seed,
    )

    params["create_datetime"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(json.dumps(params, indent=2))


def _init():
    main()


if __name__ == "__main__":
    _init()

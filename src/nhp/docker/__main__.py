import logging
import os
import threading

from nhp.docker import config
from nhp.docker.run import RunWithAzureStorage, RunWithLocalStorage, parse_args


# %%
def main():
    """the main method"""

    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.local_storage:
        runner = RunWithLocalStorage(args.params_file)
    else:
        runner = RunWithAzureStorage(args.params_file, config.APP_VERSION)

    logging.info("running model for: %s", args.params_file)
    logging.info("container timeout: %ds", config.CONTAINER_TIMEOUT_SECONDS)
    logging.info("submitted by: %s", runner.params.get("user"))
    logging.info("model_runs:   %s", runner.params["model_runs"])
    logging.info("start_year:   %s", runner.params["start_year"])
    logging.info("end_year:     %s", runner.params["end_year"])
    logging.info("app_version:  %s", runner.params["app_version"])

    saved_files, results_file = run_all(
        runner.params, "data", runner.progress_callback, args.save_full_model_results
    )

    runner.finish(results_file, saved_files, args.save_full_model_results)

    logging.info("complete")


# %%
def _exit_container():
    logging.error("\nTimed out, killing container")
    os._exit(1)


# %%
def init():
    """method for calling main"""
    if __name__ == "__main__":
        # start a timer to kill the container if we reach a timeout
        t = threading.Timer(config.CONTAINER_TIMEOUT_SECONDS, _exit_container)
        t.start()
        # run the model
        main()
        # cancel the timer
        t.cancel()


init()

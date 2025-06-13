import logging
import os
import threading

from nhp.docker import config
from nhp.docker.run import main


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


# %%
init()

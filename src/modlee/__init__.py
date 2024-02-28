""" 
Modlee package.
"""
import traceback
import importlib
import glob
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import os
from os import devnull
from os.path import dirname, basename, isfile, join

import pathlib
from pathlib import Path
from urllib.parse import urlparse

import logging, warnings


import mlflow
from mlflow import start_run

from .client import ModleeClient
api_key = os.environ.get("MODLEE_API_KEY", None)
modlee_client = ModleeClient(api_key=api_key)
from .retriever import *
from .utils import save_run
from .model_text_converter import get_code_text, get_code_text_for_model
from . import model_text_converter, exp_loss_logger, data_mf, model, recommender

logging.basicConfig(encoding="utf-8", level=logging.WARNING)
api_modules = ["model_text_converter", "exp_loss_logger"]
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]

for _logger in [
    "pytorch_lightning",
    "lightning.pytorch.core",
    "mlflow",
    "torchvision",
    "torch.nn",
]:
    pl_logger = logging.getLogger(_logger)
    pl_logger.propagate = False
    pl_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*turn shuffling off.*")
warnings.filterwarnings("ignore", ".*Arguments other than a weight enum or.*")
warnings.filterwarnings("ignore", ".*The parameter 'pretrained' is deprecated since.*")
warnings.filterwarnings("ignore", ".*Using a target size.*")
warnings.filterwarnings("ignore", ".*Implicit dimension choice.*")
warnings.filterwarnings("ignore", ".*divides the total loss by both.*")
warnings.filterwarnings(
    "ignore", ".*To copy construct from a tensor, it is recommended.*"
)
warnings.filterwarnings("ignore", ".*NLLLoss2d has been deprecated.*")
warnings.filterwarnings("ignore", ".*The default value of the antialias parameter.*")
warnings.filterwarnings("ignore", ".*No names were found for specified dynamic axes.*")
warnings.filterwarnings("ignore", ".*Starting from v1.9.0.*")


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


# Try to get an API key
def init(run_path=None, api_key=api_key):
    """
    Initialize package.
    Typically called at the beginning of a machine learning pipeline.
    Sets the run path where experiment assets will be stored.
    
    :param run_path: The path to the current run.
    """

    # if run_dir not provided, set to the same directory as the calling file
    if run_path is None or os.path.exists(run_path) == False:
        run_path = os.getcwd()

    set_run_path(run_path)
    auth(api_key)

def auth(api_key=None):
    """
    Fetches API functionality if the API key is valid.

    :param api_key: The user's API key, if it is not available as an environment variable.
    """
    # if api_key provided, reset modlee_client and reload API-locked modules
    if api_key:
        global modlee_client, get_code_text, get_code_text_for_model, data_mf, model_text_converter, exp_loss_logger
        modlee_client = ModleeClient(api_key=api_key)
        for _module in [data_mf, model_text_converter, exp_loss_logger]:
            importlib.reload(_module)
        if model_text_converter.module_available:
            from modlee.model_text_converter import (
                get_code_text,
                get_code_text_for_model,
            )


def set_run_path(run_path):
    """
    Set the path to the current run.
    This is where the experiment assets will be saved.

    :param run_path: The path to the current run.
    :raises FileNotFoundError: If the path does not exist, will not create the parent directories.
    :return: The tracking URI for the experiment. 
    """
    # Checking if path is absolute
    if not os.path.isabs(run_path):
        run_path = os.path.abspath(run_path)
        logging.debug(f"Setting run logs to abspath {run_path}")

    # Checking if path contains mlruns
    if "mlruns" not in run_path.split("/")[-1]:
        run_path = os.path.join(run_path, "mlruns")

    # Setting base directory and checking for existence
    run_dir_base = os.path.dirname(run_path)
    if not os.path.exists(run_dir_base):
        raise FileNotFoundError(
            f"No base directory {run_dir_base}, cannot set tracking URI"
        )

    # Setting tracking URI for mlflow
    tracking_uri = pathlib.Path(run_path).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def get_run_path():
    """
    Get the path to the current run.

    :return: The path to the current run.
    """
    return urlparse(mlflow.get_tracking_uri()).path


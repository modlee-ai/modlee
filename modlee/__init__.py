import traceback
import importlib
import glob
import os
from os.path import dirname, basename, isfile, join
import pathlib
from pathlib import Path
from urllib.parse import urlparse

import logging
logging.basicConfig(
    encoding='utf-8',
    level=logging.WARNING,
)
import warnings

import mlflow
from mlflow import start_run, \
    get_tracking_uri, set_tracking_uri
from modlee.api_client import ModleeAPIClient
modlee_client = ModleeAPIClient()
from modlee import model_text_converter
if model_text_converter.module_available:
    from modlee.model_text_converter import get_code_text, \
        get_code_text_for_model
else:
    get_code_text, get_code_text_for_model = None, None
from modlee.retriever import *
from . import data_stats, modlee_model, modlee_image_model
from . import *


modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f)
           and not f.endswith('__init__.py')]

for _logger in ['pytorch_lightning','lightning.pytorch.core','mlflow','torchvision']:
    pl_logger = logging.getLogger(_logger)
    pl_logger.propagate = False
    pl_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*turn shuffling off.*")
warnings.filterwarnings("ignore", ".*Arguments other than a weight enum or.*")
warnings.filterwarnings("ignore", ".*The parameter 'pretrained' is deprecated since.*")


def init(run_dir=None,api_key=None):
    """
    Initialize modlee
    - Set the run directory to save experiments
    - Set the API key
    """
    
    # if run_dir not provided, set to the same directory as the calling file
    if run_dir is None:
        calling_file = traceback.extract_stack()[-2].filename
        run_dir = os.path.dirname(calling_file)
    set_run_dir(run_dir)
    
    # if api_key provided, reset modlee_client and reload API-locked modules
    if api_key:
        global modlee_client, get_code_text, get_code_text_for_model, data_stats, model_text_converter
        modlee_client = ModleeAPIClient(
            api_key=api_key
            )
        for _module in [data_stats, model_text_converter]:
            importlib.reload(_module)
        if model_text_converter.module_available:
            from modlee.model_text_converter import get_code_text, \
                get_code_text_for_model


def set_run_dir(run_dir):
    # Checking if path is absolute
    if not os.path.isabs(run_dir):
        run_dir = os.path.abspath(run_dir)
        logging.debug(f"Setting run logs to abspath {run_dir}")
    
    # Checking if path contains mlruns
    if 'mlruns' not in run_dir.split('/')[-1]:
        run_dir = os.path.join(run_dir, 'mlruns')

    # Setting base directory and checking for existence
    run_dir_base = os.path.dirname(run_dir) 
    if not os.path.exists(run_dir_base):
        raise FileNotFoundError(
            f"No base directory {run_dir_base}, cannot set tracking URI")

    # Setting tracking URI for mlflow
    mlflow.set_tracking_uri(
        pathlib.Path(run_dir).as_uri()
    )


def get_run_dir():
    return urlparse(mlflow.get_tracking_uri()).path

def save_run(*args, **kwargs):
    modlee_client.save_run(*args,**kwargs)
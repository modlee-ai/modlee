import traceback
from mlflow import start_run, \
    get_tracking_uri, set_tracking_uri
import mlflow
import importlib
import glob
import os
import pathlib
from pathlib import Path
from urllib.parse import urlparse


import logging
logging.basicConfig(
    encoding='utf-8',
    # level=logging.DEBUG,
    level=logging.WARNING,
)

from os.path import dirname, basename, isfile, join
from modlee.api_client import ModleeAPIClient
modlee_client = ModleeAPIClient(
    # user_id='user1'
    )
from modlee.config import MODLEE_DIR, MLRUNS_DIR
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


def init(run_dir=None,api_key=None):
    if run_dir is None:
        calling_file = traceback.extract_stack()[-2].filename
        run_dir = os.path.dirname(calling_file)
    set_run_dir(run_dir)
    if api_key:
        global modlee_client, get_code_text, get_code_text_for_model
        modlee_client = ModleeAPIClient(api_key=api_key)
        for _module in [data_stats, model_text_converter]:
            importlib.reload(_module)
        if model_text_converter.module_available:
            from modlee.model_text_converter import get_code_text, \
                get_code_text_for_model
        


def set_run_dir(run_dir):
    if not os.path.isabs(run_dir):
        run_dir = os.path.abspath(run_dir)
        logging.warning(f"Setting run logs to abspath {run_dir}")
    if 'mlruns' not in run_dir.split('/')[-1]:
        run_dir = f"{run_dir}/mlruns/"

    run_dir_base = '/'.join(run_dir.split('/')[:-1])
    if not os.path.exists(run_dir_base):
        raise FileNotFoundError(
            f"No base directory {run_dir_base}, cannot set tracking URI")

    mlflow.set_tracking_uri(
        pathlib.Path(run_dir).as_uri()
    )


def get_run_dir():
    return urlparse(mlflow.get_tracking_uri()).path


def hello_world():
    print(f'from modlee')

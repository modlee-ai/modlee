import traceback
import importlib
import glob
from contextlib import contextmanager,redirect_stderr,redirect_stdout

import os
from os import devnull
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
import modlee
from modlee.api_client import ModleeAPIClient
api_key = os.environ.get('MODLEE_API_KEY',None)
modlee_client = ModleeAPIClient(api_key=api_key)

api_modules = [
    'model_text_converter',
    'exp_loss_logger'
]
# importlib.import_module
# for api_module in api_modules:
#     globals().update({
#         api_module:importlib.import_module('.', api_module)
#     })
# from modlee import \
from . import \
    model_text_converter, \
    exp_loss_logger
if model_text_converter.module_available:
    from modlee.model_text_converter import get_code_text, \
        get_code_text_for_model
else:
    get_code_text, get_code_text_for_model = None, None
from modlee.retriever import *
from . import data_stats, modlee_model, modlee_image_model
from . import *
from . import demo
from . import recommender


modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f)
           and not f.endswith('__init__.py')]

for _logger in ['pytorch_lightning','lightning.pytorch.core','mlflow','torchvision','torch.nn']:
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
warnings.filterwarnings("ignore", ".*To copy construct from a tensor, it is recommended.*")
warnings.filterwarnings("ignore", ".*NLLLoss2d has been deprecated.*")
warnings.filterwarnings("ignore", ".*The default value of the antialias parameter.*")
warnings.filterwarnings("ignore", ".*No names were found for specified dynamic axes.*")
warnings.filterwarnings("ignore", ".*Starting from v1.9.0.*")

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# Try to get an API key
def init(run_dir=None,api_key=api_key):
    """
    Initialize modlee
    - Set the run directory to save experiments
    - Set the API key
    """
    
    # if run_dir not provided, set to the same directory as the calling file
    if run_dir is None:
        calling_file = traceback.extract_stack()[-2].filename
        run_dir = os.path.dirname(calling_file)

    if os.path.exists(run_dir)==False:
        run_dir = os.getcwd()

    set_run_dir(run_dir)
    
    # if api_key provided, reset modlee_client and reload API-locked modules
    if api_key:
        global modlee_client, get_code_text, get_code_text_for_model, data_stats, model_text_converter, exp_loss_logger
        modlee_client = ModleeAPIClient(
            api_key=api_key
            )
        for _module in [data_stats, model_text_converter, exp_loss_logger]:
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
    tracking_uri = pathlib.Path(run_dir).as_uri()
    mlflow.set_tracking_uri(
        tracking_uri
    )
    return tracking_uri

def get_run_dir():
    return urlparse(mlflow.get_tracking_uri()).path

def save_run(*args, **kwargs):
    modlee_client.save_run(*args,**kwargs)



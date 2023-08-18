import importlib
import glob
import os
import pathlib
from os.path import dirname, basename, isfile, join
from modlee.config import MODLEE_DIR, MLRUNS_DIR
from modlee.model_text_converter import get_code_text, \
    get_code_text_for_model

import logging
logging.basicConfig(
    encoding='utf-8',
    # level=logging.DEBUG,
    level=logging.WARNING,
    )

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

import mlflow
from mlflow import start_run, \
    get_tracking_uri, set_tracking_uri
set_tracking_uri(f"file://{MLRUNS_DIR}")

def set_run_dir(run_dir):
    if not os.path.isabs(run_dir):
        logging.warning(f"Cannot set run logs to relative path {run_dir}; path must be absolute (os.path.abspath)")
        return 
    if 'mlruns' not in run_dir.split('/')[-1]:
        run_dir = f"{run_dir}/mlruns/"
    mlflow.set_tracking_uri(
        pathlib.Path(run_dir).as_uri()
    )
        
def hello_world():
    print(f'from modlee')

# modlee_files = sorted(glob.glob(str(MODLEE_DIR / '*.py')))
# for file in modlee_files:
#     if '__init__' in file: continue
#     mod_name = basename(file).split('.')[0]
#     mod = importlib.import_module(f'modlee.{mod_name}')
#     globals().update({mod_name:mod})
#     globals().update({v:getattr(mod,v) for v in dir(mod) if '__' not in v})



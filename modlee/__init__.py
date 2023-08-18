import importlib
import glob
from os.path import dirname, basename, isfile, join
from modlee.config import modlee_DIR
from modlee.model_text_converter import get_code_text, \
    get_code_text_for_model
from mlflow import start_run
        
import logging
logging.basicConfig(
    encoding='utf-8',
    # level=logging.DEBUG,
    level=logging.WARNING,
    )

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]


def hello_world():
    print(f'from modleee')

# modlee_files = sorted(glob.glob(str(modlee_DIR / '*.py')))
# for file in modlee_files:
#     if '__init__' in file: continue
#     mod_name = basename(file).split('.')[0]
#     mod = importlib.import_module(f'modlee.{mod_name}')
#     globals().update({mod_name:mod})
#     globals().update({v:getattr(mod,v) for v in dir(mod) if '__' not in v})



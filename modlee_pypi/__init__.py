import importlib
import glob
from os.path import basename
from modlee_pypi.config import PYPI_DIR
from modlee_pypi import data_stats, modlee_model
from modlee_pypi.model_summary import convert_model_to_text, \
    convert_text_to_model
from modlee_pypi.model_text_converter import *
from modlee_pypi.model_text_converter import get_code_text, \
    get_code_text_for_model
    
import logging
logging.basicConfig(
    encoding='utf-8',
    # level=logging.DEBUG,
    level=logging.WARNING,
    )


# pypi_files = sorted(glob.glob(str(PYPI_DIR / '*.py')))
# for file in pypi_files:
#     if '__init__' in file: continue
#     mod_name = basename(file).split('.')[0]
#     mod = importlib.import_module(f'modlee_pypi.{mod_name}')
#     globals().update({mod_name:mod})
#     globals().update({v:getattr(mod,v) for v in dir(mod) if '__' not in v})



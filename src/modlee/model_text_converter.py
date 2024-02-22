""" 
Converter for model objects to text.
"""
import inspect
import torch
import lightning.pytorch as pl
from modlee import modlee_client

_module = modlee_client.get_module("model_text_converter")
module_available = False
get_code_text, get_code_text_for_model = None, None
if _module is not None:
    exec(_module, globals())
    module_available = True

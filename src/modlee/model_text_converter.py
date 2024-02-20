""" 
Converter for model objects to text.
"""
import inspect
import torch
import lightning.pytorch as pl
from modlee import modlee_client

_module = modlee_client.get_module("model_text_converter")
module_available = False
if _module is not None:
    exec(_module, globals())
    module_available = True

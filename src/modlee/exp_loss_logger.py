""" 
Logger for different loss calculations.
"""
import lightning.pytorch as pl
from modlee import modlee_client
import logging

# Code for API client for exp_loss_logger.py
_module = modlee_client.get_module("exp_loss_logger")
module_available = False
if _module is not None:
    exec(_module, globals())
    module_available = True

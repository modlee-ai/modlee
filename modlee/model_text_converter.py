import torchvision
import inspect
import torch.nn as nn
import torch
import lightning.pytorch as pl

from modlee.api_client import ModleeAPIClient
client = ModleeAPIClient()
# globals().update(client.get_module('model_text_converter'))
_module = client.get_module('model_text_converter')
module_available = False
if _module is not None:
    exec(_module,globals())
    module_available = True

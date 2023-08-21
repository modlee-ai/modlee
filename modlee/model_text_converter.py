import torchvision
import inspect
import torch.nn as nn
import torch
import lightning.pytorch as pl

from modlee.api_client import ModleeAPIClient
client = ModleeAPIClient()
# globals().update(client.get_module('model_text_converter'))
exec(client.get_module('model_text_converter'),globals())

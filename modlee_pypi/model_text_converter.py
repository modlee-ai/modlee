import torchvision
import inspect
import torch.nn as nn
import torch
import lightning.pytorch as pl


modlee_required_packages = '''
import torch
import torch.nn as nn
import pytorch_lightning
import pytorch_lightning as pl
from torch.nn import functional as F
import lightning

import torch
import torch.nn as nn
import inspect

import numpy
import numpy as np

try:
    from torchmetrics.functional import accuracy
except ImportError:
    from pytorch_lightning.metrics.functional import accuracy

#ptorch out-of-box libs needed
from torchvision.utils import _log_api_usage_once

from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision import models
import torchvision

# from torchvision.ops.misc import Conv2dNormActivation, Permute
# from torchvision.ops.stochastic_depth import StochasticDepth
# from torchvision.transforms._presets import ImageClassification
# from torchvision.utils import _log_api_usage_once
# from torchvision.models._api import register_model, Weights, WeightsEnum
# from torchvision.models._meta import _IMAGENET_CATEGORIES
# from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

# from torchvision.models.convnext import *
# from torchvision.models.convnext import CNBlockConfig

# Modlee imports
import modlee_pypi

'''

ops_list = [set(nn.__dir__()), set(torchvision.models.__dir__())]
ops_merged = set().union(*ops_list)

class_header = 'class {}({}):\n'

modlee_model_name = 'ModleeModel'


def exhaust_sequence_branch(root_sequence_module, custom_history):
    """
    exhaust_sequence_branch _summary_

    Args:
        root_sequence_module (_type_): _description_
        custom_history (_type_): _description_

    Returns:
        _type_: _description_
    """

    sequences = [root_sequence_module]
    custom_modules_list = []

    while len(sequences) != 0:
        sequence = sequences[-1]
        module_list = list(sequence.__dict__['_modules'].values())
        for module in module_list:
            module_name = str(module).split('(')[0]
            if module_name not in ops_merged and module_name not in custom_history:
                print('found custom module: ', module_name)
                custom_modules_list.append(module)
                custom_history.add(module_name)
            elif module_name == 'Sequential':
                print('found sequential go as deep as needed')
                sequences.insert(0, module)
        sequences.pop()

    return custom_modules_list, custom_history


def get_code_text(code_text, module, custom_history):
    """
    get_code_text _summary_
    
    TODO - this only saves a hardcoded subset of functions

    Args:
        code_text (_type_): _description_
        module (_type_): _description_
        custom_history (_type_): _description_
    """
    # get current and parent class names
    module_class_name = module.__class__.__name__
    print('module_class_name: ', module_class_name)
    module_parent_class_name = str(module.__class__.__bases__[0]).split("'")[1]
    print('module_parent_class_name: ', module_parent_class_name)
    # format the class header
    class_header_code = class_header.format(
        module_class_name, module_parent_class_name)

    forward_code = inspect.getsource(module.forward)
    init_code = inspect.getsource(module.__init__)
    
    functions_to_save = ['__init__','forward']
    if module_parent_class_name.split('.')[-1] in ['LightningModule','ModleeModel']:
        functions_to_save += ['training_step','validation_step','configure_optimizers']
    function_code = {}
    for function_to_save in functions_to_save:
        _function = getattr(module,function_to_save,None)
        if _function is None:
            functions_to_save.remove(function_to_save)
            continue
        function_code.update({
            function_to_save:inspect.getsource(_function)
        })

    if code_text == '':
        class_header_code = class_header_code.replace(
            module_class_name, modlee_model_name)
        # init_code = init_code.replace(module_class_name, modlee_model_name)
        function_code['__init__'] = function_code['__init__'].replace(
            module_class_name, modlee_model_name)

    # exclude modlee_required_packages from training data? if it is the same ...
    # code_text = code_text + class_header_code + init_code + forward_code + '\n'
    
    # code_text = code_text + class_header
    # code
    code_text = '\n'.join([
        code_text, class_header_code,
        ] + list(function_code.values()) + ['\n'])

    # -------------

    child_module_list = list(module.__dict__['_modules'].values())

    custom_child_module_list = []

    for child_module in child_module_list:
        child_module_name = str(child_module).split('(')[0]
        print(child_module_name)
        if child_module_name not in ops_merged and child_module_name not in custom_history:
            print('found custom module: ', child_module_name)
            custom_child_module_list.append(child_module)
            custom_history.add(child_module_name)
        elif child_module_name == 'Sequential':
            print('found sequential go as deep as needed')
            seq_custom_modules_list, custom_history = exhaust_sequence_branch(
                child_module, custom_history)
            custom_child_module_list = custom_child_module_list + seq_custom_modules_list

# NOTE THIS DOESN't solve the problem systematically, just one layer deeper ....

    return code_text, custom_child_module_list, custom_history


def get_code_text_for_model(model: nn.modules.module.Module | pl.core.module.LightningModule, include_header=False):
    """
    get_code_text_for_model _summary_

    Args:
        model (nn.modules.module.Module | pl.core.module.LightningModule): _description_

    Returns:
        _type_: _description_
    """

    code_text = ''
    custom_history = set()
    module_queue = [model]

    while len(module_queue) != 0:
        code_text, custom_child_module_list, custom_history = get_code_text(
            code_text, module_queue[-1], custom_history)
        module_queue.pop()
        module_queue = module_queue + custom_child_module_list

        print(module_queue)

    print('\n\n---code_text--- \n\n{}\n\n'.format(code_text))

    if include_header:
        code_text = modlee_required_packages + code_text
    return code_text


def save_code_text_for_model(code_text: str, include_header: bool=False):
    """
    save_code_text_for_model _summary_

    Args:
        code_text (str): _description_
    """
    if include_header: code_text = modlee_required_packages + code_text
    file_path = "modlee_model.py"  # Specify the file path

    # Open the file in write mode and write the text
    with open(file_path, 'w') as file:
        file.write(code_text)
    
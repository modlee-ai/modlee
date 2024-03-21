""" 
Convertor for model objects into text (deprecated?).
"""
import inspect
import torch.nn as nn
import torch
import lightning.pytorch as pl

module_available = True

modlee_required_packages = """
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning
import lightning.pytorch as pl

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
from typing import Any, Callable, List, Optional, Sequence, Literal

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision import models
import torchvision
import torchmetrics

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
import modlee

"""

OPS_MERGED = {
    "utils",
    "HingeEmbeddingLoss",
    "ConstantPad2d",
    "efficientnet_v2_l",
    "convnext_large",
    "GRUCell",
    "CTCLoss",
    "maxvit",
    "FeatureAlphaDropout",
    "mnasnet0_75",
    "RegNet_Y_800MF_Weights",
    "AlphaDropout",
    "Conv1d",
    "LazyConvTranspose2d",
    "__spec__",
    "swin_b",
    "RNNCellBase",
    "mobilenet_v3_small",
    "Hardsigmoid",
    "__package__",
    "GoogLeNetOutputs",
    "Dropout1d",
    "vit_b_16",
    "ConvTranspose1d",
    "LPPool1d",
    "LazyLinear",
    "efficientnet_b6",
    "grad",
    "GELU",
    "SoftMarginLoss",
    "DenseNet169_Weights",
    "Conv2d",
    "parameter",
    "ELU",
    "ReLU",
    "wide_resnet101_2",
    "__path__",
    "regnet_x_400mf",
    "PixelShuffle",
    "vgg13",
    "efficientnet_b3",
    "efficientnet_b2",
    "MNASNet1_3_Weights",
    "Softmax",
    "LazyConv1d",
    "DenseNet161_Weights",
    "MNASNet0_75_Weights",
    "__doc__",
    "RReLU",
    "SqueezeNet",
    "mobilenet_v3_large",
    "DenseNet201_Weights",
    "TripletMarginLoss",
    "AdaptiveMaxPool1d",
    "mnasnet1_3",
    "Fold",
    "efficientnet_b7",
    "efficientnet_b5",
    "VGG13_BN_Weights",
    "Conv3d",
    "Parameter",
    "googlenet",
    "MultiMarginLoss",
    "convnext_small",
    "TripletMarginWithDistanceLoss",
    "get_model",
    "modules",
    "MobileNetV2",
    "ResNet101_Weights",
    "squeezenet1_1",
    "ConvNeXt_Tiny_Weights",
    "ResNeXt50_32X4D_Weights",
    "Swin_V2_T_Weights",
    "NLLLoss",
    "LazyConvTranspose1d",
    "get_weight",
    "vgg",
    "parallel",
    "SqueezeNet1_0_Weights",
    "inception",
    "resnet",
    "swin_v2_s",
    "factory_kwargs",
    "VGG16_BN_Weights",
    "LazyBatchNorm1d",
    "EfficientNet_B7_Weights",
    "DenseNet",
    "swin_t",
    "SmoothL1Loss",
    "Dropout2d",
    "Upsample",
    "TransformerDecoder",
    "Weights",
    "EfficientNet_B4_Weights",
    "ShuffleNet_V2_X1_5_Weights",
    "LazyBatchNorm3d",
    "resnet34",
    "ConstantPad1d",
    "BCEWithLogitsLoss",
    "RegNet",
    "CELU",
    "ModuleDict",
    "mobilenet_v2",
    "swin_s",
    "densenet161",
    "squeezenet",
    "BatchNorm3d",
    "AdaptiveAvgPool3d",
    "MultiLabelSoftMarginLoss",
    "regnet_x_3_2gf",
    "init",
    "__builtins__",
    "resnet18",
    "MobileNet_V2_Weights",
    "ShuffleNetV2",
    "Swin_V2_S_Weights",
    "ShuffleNet_V2_X0_5_Weights",
    "functional",
    "resnext50_32x4d",
    "optical_flow",
    "VisionTransformer",
    "AdaptiveLogSoftmaxWithLoss",
    "UninitializedBuffer",
    "MaxUnpool1d",
    "efficientnet_b1",
    "quantizable",
    "LazyInstanceNorm1d",
    "efficientnet",
    "VGG11_Weights",
    "LeakyReLU",
    "CosineEmbeddingLoss",
    "Hardswish",
    "RegNet_Y_16GF_Weights",
    "Linear",
    "ConvTranspose3d",
    "resnext101_64x4d",
    "RegNet_X_16GF_Weights",
    "HuberLoss",
    "inception_v3",
    "regnet_y_400mf",
    "vgg11_bn",
    "Tanhshrink",
    "Transformer",
    "GroupNorm",
    "efficientnet_v2_m",
    "ViT_L_16_Weights",
    "PoissonNLLLoss",
    "vgg16",
    "ParameterDict",
    "Softplus",
    "squeezenet1_0",
    "BatchNorm2d",
    "SELU",
    "PairwiseDistance",
    "efficientnet_b0",
    "LazyInstanceNorm3d",
    "LocalResponseNorm",
    "shufflenet_v2_x1_5",
    "vgg19",
    "__cached__",
    "swin_v2_t",
    "__name__",
    "mnasnet",
    "GLU",
    "MNASNet0_5_Weights",
    "VGG16_Weights",
    "_GoogLeNetOutputs",
    "RNNCell",
    "VGG",
    "SqueezeNet1_1_Weights",
    "ConvNeXt_Large_Weights",
    "common_types",
    "mobilenetv3",
    "GRU",
    "Mish",
    "mnasnet1_0",
    "MobileNet_V3_Large_Weights",
    "TransformerEncoder",
    "Threshold",
    "EfficientNet_V2_M_Weights",
    "Sigmoid",
    "regnet_y_800mf",
    "AdaptiveAvgPool2d",
    "Container",
    "EfficientNet_V2_L_Weights",
    "ViT_B_16_Weights",
    "SwinTransformer",
    "Softsign",
    "vit_l_16",
    "PixelUnshuffle",
    "MarginRankingLoss",
    "MaxPool2d",
    "regnet_y_16gf",
    "regnet_y_8gf",
    "densenet121",
    "ConvNeXt",
    "ReflectionPad3d",
    "mobilenetv2",
    "MSELoss",
    "RegNet_X_400MF_Weights",
    "MaxUnpool3d",
    "ShuffleNet_V2_X2_0_Weights",
    "list_models",
    "Embedding",
    "Softmin",
    "EfficientNet_B2_Weights",
    "PReLU",
    "shufflenet_v2_x0_5",
    "mobilenet",
    "ConvNeXt_Base_Weights",
    "Swin_S_Weights",
    "Identity",
    "InstanceNorm3d",
    "resnet152",
    "Unflatten",
    "_utils",
    "ViT_B_32_Weights",
    "efficientnet_v2_s",
    "ReflectionPad2d",
    "Hardshrink",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "MultiLabelMarginLoss",
    "CrossMapLRN2d",
    "VGG13_Weights",
    "AvgPool3d",
    "RegNet_Y_32GF_Weights",
    "shufflenet_v2_x1_0",
    "LazyConv3d",
    "LSTM",
    "vgg11",
    "WeightsEnum",
    "MultiheadAttention",
    "swin_transformer",
    "get_model_builder",
    "regnet_x_800mf",
    "detection",
    "RegNet_X_32GF_Weights",
    "Wide_ResNet50_2_Weights",
    "MaxVit",
    "AlexNet",
    "MaxPool1d",
    "SiLU",
    "ShuffleNet_V2_X1_0_Weights",
    "GoogLeNet_Weights",
    "AdaptiveMaxPool2d",
    "maxvit_t",
    "BCELoss",
    "DataParallel",
    "Tanh",
    "EfficientNet_V2_S_Weights",
    "MobileNetV3",
    "EfficientNet_B3_Weights",
    "get_model_weights",
    "ReflectionPad1d",
    "intrinsic",
    "ParameterList",
    "ReplicationPad1d",
    "EfficientNet_B5_Weights",
    "ConvNeXt_Small_Weights",
    "qat",
    "densenet169",
    "vgg19_bn",
    "swin_v2_b",
    "RegNet_X_8GF_Weights",
    "EmbeddingBag",
    "ConvTranspose2d",
    "vit_b_32",
    "NLLLoss2d",
    "vision_transformer",
    "KLDivLoss",
    "__file__",
    "Module",
    "EfficientNet_B6_Weights",
    "GaussianNLLLoss",
    "TransformerEncoderLayer",
    "BatchNorm1d",
    "Flatten",
    "regnet_y_3_2gf",
    "ReplicationPad3d",
    "VGG19_BN_Weights",
    "resnet50",
    "L1Loss",
    "wide_resnet50_2",
    "convnext",
    "UpsamplingNearest2d",
    "vgg16_bn",
    "regnet_y_128gf",
    "_api",
    "LPPool2d",
    "convnext_tiny",
    "Swin_B_Weights",
    "ConstantPad3d",
    "RegNet_Y_8GF_Weights",
    "AvgPool2d",
    "resnext101_32x8d",
    "UninitializedParameter",
    "AvgPool1d",
    "video",
    "segmentation",
    "MNASNet",
    "mnasnet0_5",
    "regnet_x_32gf",
    "convnext_base",
    "regnet_x_8gf",
    "LogSoftmax",
    "VGG11_BN_Weights",
    "ViT_L_32_Weights",
    "ZeroPad2d",
    "regnet_x_1_6gf",
    "regnet_y_32gf",
    "RNNBase",
    "GoogLeNet",
    "Swin_T_Weights",
    "LayerNorm",
    "__loader__",
    "AlexNet_Weights",
    "RegNet_X_800MF_Weights",
    "Inception3",
    "RNN",
    "_InceptionOutputs",
    "FractionalMaxPool3d",
    "vgg13_bn",
    "Dropout",
    "CrossEntropyLoss",
    "vit_h_14",
    "InceptionOutputs",
    "RegNet_Y_128GF_Weights",
    "InstanceNorm2d",
    "MaxUnpool2d",
    "_reduction",
    "LogSigmoid",
    "efficientnet_b4",
    "quantization",
    "ResNet",
    "resnet101",
    "LazyBatchNorm2d",
    "ChannelShuffle",
    "Softmax2d",
    "TransformerDecoderLayer",
    "LazyConv2d",
    "MNASNet1_0_Weights",
    "FractionalMaxPool2d",
    "shufflenetv2",
    "EfficientNet",
    "MaxPool3d",
    "SyncBatchNorm",
    "Softshrink",
    "MaxVit_T_Weights",
    "RegNet_X_3_2GF_Weights",
    "ResNet18_Weights",
    "Unfold",
    "ViT_H_14_Weights",
    "AdaptiveAvgPool1d",
    "MobileNet_V3_Small_Weights",
    "VGG19_Weights",
    "RegNet_Y_400MF_Weights",
    "ModuleList",
    "_meta",
    "ResNet34_Weights",
    "Dropout3d",
    "LSTMCell",
    "AdaptiveMaxPool3d",
    "Inception_V3_Weights",
    "EfficientNet_B0_Weights",
    "Hardtanh",
    "RegNet_Y_3_2GF_Weights",
    "LazyInstanceNorm2d",
    "EfficientNet_B1_Weights",
    "DenseNet121_Weights",
    "regnet_x_16gf",
    "Wide_ResNet101_2_Weights",
    "RegNet_X_1_6GF_Weights",
    "quantized",
    "CosineSimilarity",
    "ReplicationPad2d",
    "Bilinear",
    "densenet",
    "RegNet_Y_1_6GF_Weights",
    "regnet_y_1_6gf",
    "ReLU6",
    "InstanceNorm1d",
    "densenet201",
    "LazyConvTranspose3d",
    "alexnet",
    "vit_l_32",
    "shufflenet_v2_x2_0",
    "Sequential",
    "Swin_V2_B_Weights",
    "regnet",
    "ResNet50_Weights",
    "ResNet152_Weights",
    "UpsamplingBilinear2d",
}
class_header = "class {}({}):\n"

modlee_model_name = "ModleeModel"


def exhaust_sequence_branch(root_sequence_module, custom_history):
    """
    Exhaust a module as a tree to find custom modules,

    :param root_sequence_module: The root node/module of the model.
    :param custom_history: A list of custom module names.
    :return: A tuple of lists ([custom module objects], [custom module names])
    """

    sequences = [root_sequence_module]
    custom_modules_list = []

    while len(sequences) != 0:
        sequence = sequences[-1]
        module_list = list(sequence.__dict__["_modules"].values())
        for module in module_list:
            module_name = str(module).split("(")[0]
            if module_name not in OPS_MERGED and module_name not in custom_history:
                # print('found custom module: ', module_name)
                custom_modules_list.append(module)
                custom_history.add(module_name)
            elif module_name == "Sequential":
                # print('found sequential go as deep as needed')
                sequences.insert(0, module)
        sequences.pop()

    return custom_modules_list, custom_history


def get_code_text(code_text, module, custom_history):
    """
    Get a code text representation of a model as its __init__ and forward functions.
    TODO - this only saves a hardcoded subset of functions

    :param code_text: The current code text if there are other dependencies, can be empty ''.
    :param module: The module for which to get the code text.
    :param custom_history: A list of the history of custom modules.
    :return: A tuple of code_text, custom_child_module_list, custom_history.
    """
    # get current and parent class names
    module_class_name = module.__class__.__name__
    module_parent_class_name = str(module.__class__.__bases__[0]).split("'")[1]
    # format the class header
    class_header_code = class_header.format(module_class_name, module_parent_class_name)

    forward_code = inspect.getsource(module.forward)
    init_code = inspect.getsource(module.__init__)

    functions_to_save = ["__init__", "forward"]
    if module_parent_class_name.split(".")[-1] in ["LightningModule", "ModleeModel"]:
        functions_to_save += [
            "training_step",
            "validation_step",
            "configure_optimizers",
        ]
    function_code = {}
    for function_to_save in functions_to_save:
        _function = getattr(module, function_to_save, None)
        if _function is None:
            functions_to_save.remove(function_to_save)
            continue
        # elif _function==getattr(pl.LightningModule,function_to_save):

        # elif not pl.utilities.model_helpers.is_overridden('validation_step',module):
        #     functions_to_save.remove(function_to_save)
        #     continue

        function_code.update({function_to_save: inspect.getsource(_function)})

    if code_text == "":
        class_header_code = class_header_code.replace(
            module_class_name, modlee_model_name
        )
        # init_code = init_code.replace(module_class_name, modlee_model_name)
        function_code["__init__"] = function_code["__init__"].replace(
            module_class_name, modlee_model_name
        )

    # exclude modlee_required_packages from training data? if it is the same ...
    # code_text = code_text + class_header_code + init_code + forward_code + '\n'

    # code_text = code_text + class_header
    # code
    code_text = "\n".join(
        [code_text, class_header_code] + list(function_code.values()) + ["\n"]
    )

    # -------------

    child_module_list = list(module.__dict__["_modules"].values())

    custom_child_module_list = []

    for child_module in child_module_list:
        child_module_name = str(child_module).split("(")[0]
        # print(child_module_name)
        if (
            child_module_name not in OPS_MERGED
            and child_module_name not in custom_history
        ):
            # print('found custom module: ', child_module_name)
            custom_child_module_list.append(child_module)
            custom_history.add(child_module_name)
        elif child_module_name == "Sequential":
            # print('found sequential go as deep as needed')
            seq_custom_modules_list, custom_history = exhaust_sequence_branch(
                child_module, custom_history
            )
            custom_child_module_list = (
                custom_child_module_list + seq_custom_modules_list
            )

    # NOTE THIS DOESN't solve the problem systematically, just one layer deeper ....

    return code_text, custom_child_module_list, custom_history


def get_code_text_for_model(
    model: nn.modules.module.Module | pl.core.module.LightningModule,
    include_header=False,
):
    """
    Get the code text for a model.

    :param model: The model for which to get the code text.
    :return: The code text for the model.
    """

    code_text = ""
    custom_history = set()
    module_queue = [model]

    while len(module_queue) != 0:
        code_text, custom_child_module_list, custom_history = get_code_text(
            code_text, module_queue[-1], custom_history
        )
        module_queue.pop()
        module_queue = module_queue + custom_child_module_list

        # print(module_queue)

    # print('\n\n---code_text--- \n\n{}\n\n'.format(code_text))

    if include_header:
        code_text = modlee_required_packages + code_text
    return code_text


def save_code_text_for_model(code_text: str, include_header: bool = False):
    """
    Save the code text.

    :param code_text: The code text to save.
    :param include_header: Whether to include the header of modlee imports in the text, defaults to False.
    """
    if include_header:
        code_text = modlee_required_packages + code_text
    file_path = "modlee_model.py"  # Specify the file path

    # Open the file in write mode and write the text
    with open(file_path, "w") as file:
        file.write(code_text)

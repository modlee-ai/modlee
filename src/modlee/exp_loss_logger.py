""" 
Log several different losses for an experiment.
"""
import inspect
import ast
import textwrap
import logging
import os
from _ast import ClassDef

module_available = True
OPS_MERGED = {'utils', 'HingeEmbeddingLoss', 'ConstantPad2d', 'efficientnet_v2_l', 'convnext_large', 'GRUCell', 'CTCLoss', 'maxvit', 'FeatureAlphaDropout', 'mnasnet0_75', 'RegNet_Y_800MF_Weights', 'AlphaDropout', 'Conv1d', 'LazyConvTranspose2d', '__spec__', 'swin_b', 'RNNCellBase', 'mobilenet_v3_small', 'Hardsigmoid', '__package__', 'GoogLeNetOutputs', 'Dropout1d', 'vit_b_16', 'ConvTranspose1d', 'LPPool1d', 'LazyLinear', 'efficientnet_b6', 'grad', 'GELU', 'SoftMarginLoss', 'DenseNet169_Weights', 'Conv2d', 'parameter', 'ELU', 'ReLU', 'wide_resnet101_2', '__path__', 'regnet_x_400mf', 'PixelShuffle', 'vgg13', 'efficientnet_b3', 'efficientnet_b2', 'MNASNet1_3_Weights', 'Softmax', 'LazyConv1d', 'DenseNet161_Weights', 'MNASNet0_75_Weights', '__doc__', 'RReLU', 'SqueezeNet', 'mobilenet_v3_large', 'DenseNet201_Weights', 'TripletMarginLoss', 'AdaptiveMaxPool1d', 'mnasnet1_3', 'Fold', 'efficientnet_b7', 'efficientnet_b5', 'VGG13_BN_Weights', 'Conv3d', 'Parameter', 'googlenet', 'MultiMarginLoss', 'convnext_small', 'TripletMarginWithDistanceLoss', 'get_model', 'modules', 'MobileNetV2', 'ResNet101_Weights', 'squeezenet1_1', 'ConvNeXt_Tiny_Weights', 'ResNeXt50_32X4D_Weights', 'Swin_V2_T_Weights', 'NLLLoss', 'LazyConvTranspose1d', 'get_weight', 'vgg', 'parallel', 'SqueezeNet1_0_Weights', 'inception', 'resnet', 'swin_v2_s', 'factory_kwargs', 'VGG16_BN_Weights', 'LazyBatchNorm1d', 'EfficientNet_B7_Weights', 'DenseNet', 'swin_t', 'SmoothL1Loss', 'Dropout2d', 'Upsample', 'TransformerDecoder', 'Weights', 'EfficientNet_B4_Weights', 'ShuffleNet_V2_X1_5_Weights', 'LazyBatchNorm3d', 'resnet34', 'ConstantPad1d', 'BCEWithLogitsLoss', 'RegNet', 'CELU', 'ModuleDict', 'mobilenet_v2', 'swin_s', 'densenet161', 'squeezenet', 'BatchNorm3d', 'AdaptiveAvgPool3d', 'MultiLabelSoftMarginLoss', 'regnet_x_3_2gf', 'init', '__builtins__', 'resnet18', 'MobileNet_V2_Weights', 'ShuffleNetV2', 'Swin_V2_S_Weights', 'ShuffleNet_V2_X0_5_Weights', 'functional', 'resnext50_32x4d', 'optical_flow', 'VisionTransformer', 'AdaptiveLogSoftmaxWithLoss', 'UninitializedBuffer', 'MaxUnpool1d', 'efficientnet_b1', 'quantizable', 'LazyInstanceNorm1d', 'efficientnet', 'VGG11_Weights', 'LeakyReLU', 'CosineEmbeddingLoss', 'Hardswish', 'RegNet_Y_16GF_Weights', 'Linear', 'ConvTranspose3d', 'resnext101_64x4d', 'RegNet_X_16GF_Weights', 'HuberLoss', 'inception_v3', 'regnet_y_400mf', 'vgg11_bn', 'Tanhshrink', 'Transformer', 'GroupNorm', 'efficientnet_v2_m', 'ViT_L_16_Weights', 'PoissonNLLLoss', 'vgg16', 'ParameterDict', 'Softplus', 'squeezenet1_0', 'BatchNorm2d', 'SELU', 'PairwiseDistance', 'efficientnet_b0', 'LazyInstanceNorm3d', 'LocalResponseNorm', 'shufflenet_v2_x1_5', 'vgg19', '__cached__', 'swin_v2_t', '__name__', 'mnasnet', 'GLU', 'MNASNet0_5_Weights', 'VGG16_Weights', '_GoogLeNetOutputs', 'RNNCell', 'VGG', 'SqueezeNet1_1_Weights', 'ConvNeXt_Large_Weights', 'common_types', 'mobilenetv3', 'GRU', 'Mish', 'mnasnet1_0', 'MobileNet_V3_Large_Weights', 'TransformerEncoder', 'Threshold', 'EfficientNet_V2_M_Weights', 'Sigmoid', 'regnet_y_800mf', 'AdaptiveAvgPool2d', 'Container', 'EfficientNet_V2_L_Weights', 'ViT_B_16_Weights', 'SwinTransformer', 'Softsign', 'vit_l_16', 'PixelUnshuffle', 'MarginRankingLoss', 'MaxPool2d', 'regnet_y_16gf', 'regnet_y_8gf', 'densenet121', 'ConvNeXt', 'ReflectionPad3d', 'mobilenetv2', 'MSELoss', 'RegNet_X_400MF_Weights', 'MaxUnpool3d',
              'ShuffleNet_V2_X2_0_Weights', 'list_models', 'Embedding', 'Softmin', 'EfficientNet_B2_Weights', 'PReLU', 'shufflenet_v2_x0_5', 'mobilenet', 'ConvNeXt_Base_Weights', 'Swin_S_Weights', 'Identity', 'InstanceNorm3d', 'resnet152', 'Unflatten', '_utils', 'ViT_B_32_Weights', 'efficientnet_v2_s', 'ReflectionPad2d', 'Hardshrink', 'ResNeXt101_32X8D_Weights', 'ResNeXt101_64X4D_Weights', 'MultiLabelMarginLoss', 'CrossMapLRN2d', 'VGG13_Weights', 'AvgPool3d', 'RegNet_Y_32GF_Weights', 'shufflenet_v2_x1_0', 'LazyConv3d', 'LSTM', 'vgg11', 'WeightsEnum', 'MultiheadAttention', 'swin_transformer', 'get_model_builder', 'regnet_x_800mf', 'detection', 'RegNet_X_32GF_Weights', 'Wide_ResNet50_2_Weights', 'MaxVit', 'AlexNet', 'MaxPool1d', 'SiLU', 'ShuffleNet_V2_X1_0_Weights', 'GoogLeNet_Weights', 'AdaptiveMaxPool2d', 'maxvit_t', 'BCELoss', 'DataParallel', 'Tanh', 'EfficientNet_V2_S_Weights', 'MobileNetV3', 'EfficientNet_B3_Weights', 'get_model_weights', 'ReflectionPad1d', 'intrinsic', 'ParameterList', 'ReplicationPad1d', 'EfficientNet_B5_Weights', 'ConvNeXt_Small_Weights', 'qat', 'densenet169', 'vgg19_bn', 'swin_v2_b', 'RegNet_X_8GF_Weights', 'EmbeddingBag', 'ConvTranspose2d', 'vit_b_32', 'NLLLoss2d', 'vision_transformer', 'KLDivLoss', '__file__', 'Module', 'EfficientNet_B6_Weights', 'GaussianNLLLoss', 'TransformerEncoderLayer', 'BatchNorm1d', 'Flatten', 'regnet_y_3_2gf', 'ReplicationPad3d', 'VGG19_BN_Weights', 'resnet50', 'L1Loss', 'wide_resnet50_2', 'convnext', 'UpsamplingNearest2d', 'vgg16_bn', 'regnet_y_128gf', '_api', 'LPPool2d', 'convnext_tiny', 'Swin_B_Weights', 'ConstantPad3d', 'RegNet_Y_8GF_Weights', 'AvgPool2d', 'resnext101_32x8d', 'UninitializedParameter', 'AvgPool1d', 'video', 'segmentation', 'MNASNet', 'mnasnet0_5', 'regnet_x_32gf', 'convnext_base', 'regnet_x_8gf', 'LogSoftmax', 'VGG11_BN_Weights', 'ViT_L_32_Weights', 'ZeroPad2d', 'regnet_x_1_6gf', 'regnet_y_32gf', 'RNNBase', 'GoogLeNet', 'Swin_T_Weights', 'LayerNorm', '__loader__', 'AlexNet_Weights', 'RegNet_X_800MF_Weights', 'Inception3', 'RNN', '_InceptionOutputs', 'FractionalMaxPool3d', 'vgg13_bn', 'Dropout', 'CrossEntropyLoss', 'vit_h_14', 'InceptionOutputs', 'RegNet_Y_128GF_Weights', 'InstanceNorm2d', 'MaxUnpool2d', '_reduction', 'LogSigmoid', 'efficientnet_b4', 'quantization', 'ResNet', 'resnet101', 'LazyBatchNorm2d', 'ChannelShuffle', 'Softmax2d', 'TransformerDecoderLayer', 'LazyConv2d', 'MNASNet1_0_Weights', 'FractionalMaxPool2d', 'shufflenetv2', 'EfficientNet', 'MaxPool3d', 'SyncBatchNorm', 'Softshrink', 'MaxVit_T_Weights', 'RegNet_X_3_2GF_Weights', 'ResNet18_Weights', 'Unfold', 'ViT_H_14_Weights', 'AdaptiveAvgPool1d', 'MobileNet_V3_Small_Weights', 'VGG19_Weights', 'RegNet_Y_400MF_Weights', 'ModuleList', '_meta', 'ResNet34_Weights', 'Dropout3d', 'LSTMCell', 'AdaptiveMaxPool3d', 'Inception_V3_Weights', 'EfficientNet_B0_Weights', 'Hardtanh', 'RegNet_Y_3_2GF_Weights', 'LazyInstanceNorm2d', 'EfficientNet_B1_Weights', 'DenseNet121_Weights', 'regnet_x_16gf', 'Wide_ResNet101_2_Weights', 'RegNet_X_1_6GF_Weights', 'quantized', 'CosineSimilarity', 'ReplicationPad2d', 'Bilinear', 'densenet', 'RegNet_Y_1_6GF_Weights', 'regnet_y_1_6gf', 'ReLU6', 'InstanceNorm1d', 'densenet201', 'LazyConvTranspose3d', 'alexnet', 'vit_l_32', 'shufflenet_v2_x2_0', 'Sequential', 'Swin_V2_B_Weights', 'regnet', 'ResNet50_Weights', 'ResNet152_Weights', 'UpsamplingBilinear2d'}
class_header = 'class {}({}):\n'

modlee_model_name = 'ModleeModel'


def extract_loss_functions(code_str: str):
    """
    Extracts unique loss function calls from ModleeModel class definition passed as string

    :param code_str: The Python code string to analyze.
    :return: A list of unique loss function calls found in the code.
    """
    # Parse the code string into an Abstract Syntax Tree (AST)
    code_str = '\n'+textwrap.dedent(code_str)

    tree = ast.parse(code_str)
    # Initialize variables to store information
    loss_function_calls = []  # List to store identified loss function calls
    base_class = None  # Variable to store the base class of an instantiated alias

    # Define a visitor to traverse the AST
    class LossFunctionCallVisitor(ast.NodeVisitor):
        """ 
        Helper class to determine the loss functions used.
        """
        def visit_ClassDef(self, node: ClassDef):
            # Check if this class is a ModleeModel
            if isinstance(node, ast.ClassDef) and node.name == 'ModleeModel':
                # Check all function calls in this class
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        # Check if it's a loss function call
                        if (
                            isinstance(child.func, ast.Attribute) and
                            isinstance(child.func.value, ast.Name) and
                            isinstance(child.func.attr, str) and
                            (child.func.value.id == 'nn' or child.func.value.id == 'F')
                        ):
                            loss_function_calls.append(child.func.attr)

    # Instantiate the visitor
    visitor = LossFunctionCallVisitor()

    # Visit the AST to collect loss function calls
    visitor.visit(tree)

    return list(set(loss_function_calls))



def get_exp_loss_for_model(module):
    """
    Get a list of loss functions used in a given model.
    NOTE Assumption that loss function is called in __init__ or training_step

    :param module: The model to analyze.
    :return: A list of unique loss function calls found in the model.
    """

    # Get the current and parent class names
    module_class_name = module.__class__.__name__
    module_parent_class_name = str(module.__class__.__bases__[0]).split("'")[1]
    
    # format the class header
    class_header_code = class_header.format(
        module_class_name, module_parent_class_name)

    # Initialize variables to store loss function information
    is_lightning_module = False
    loss_inits = ['__init__']    
    
    # Check if the module is a LightningModule or ModleeModel
    if module_parent_class_name.split('.')[-1] in ['LightningModule', 'ModleeModel']:
        loss_inits += ['training_step']
        is_lightning_module = True

    loss_calls = []

    # Iterate through the specified functions to check for loss function calls
    for function_to_save in loss_inits:
        _function = getattr(module, function_to_save, None)
        if _function is None:
            loss_inits.remove(function_to_save)
            continue
        
        function_code_str = inspect.getsource(_function)

        if is_lightning_module and function_to_save in loss_inits:
            for item in extract_loss_functions(function_code_str):
                if item!=None: 
                    loss_calls.append(item)
                     
    return list(set(loss_calls))
import random
import inspect
from torchvision import models as tvm
import wonderwords
from wonderwords import RandomWord, Defaults
import modlee
from modlee.utils import image_loaders

IMAGE_MODELS = IMAGE_CLASSIFICATION_MODELS = [
    tvm.resnet18(weights="DEFAULT"),
    tvm.resnet18(),
    tvm.resnet50(),
    tvm.resnet152(),
    # tvm.alexnet(),
    # tvm.googlenet(),
]


def get_all_image_models():
    """
    Scrapes available easily loadable torchvsion models.
    As of 240814, 63/80 are passing data metafeature calculations(?)
    Failures are due to wrong output sizes (e.g. 300 instead of 224)
        and unsupported ONNX operations.

    :return: _description_
    """
    IMAGE_MODELS = []
    for attr in dir(tvm):
        tvm_attr = getattr(tvm, attr)
        if not callable(tvm_attr) or isinstance(tvm_attr, type):
            continue
        try:
            inspect.signature(tvm_attr).bind()
        except TypeError:
            continue
        tvm_attr_ret = tvm_attr()
        if "forward" in dir(tvm_attr_ret):
            print(f"Adding {tvm_attr}")
            IMAGE_MODELS.append(tvm_attr_ret)
    return IMAGE_MODELS


IMAGE_SEGMENTATION_MODELS = [
    tvm.segmentation.fcn_resnet50(),
    tvm.segmentation.fcn_resnet101(),
    # tvm.segmentation.lraspp(),
    tvm.segmentation.lraspp_mobilenet_v3_large(),
    tvm.segmentation.deeplabv3_resnet50(),
    tvm.segmentation.deeplabv3_resnet101(),
]


"""
For use in @pytest.mark.parametrize, 
create modality-task-{kwargs,model} tuples to 
"""
IMAGE_MODALITY_TASK_KWARGS = [
    ("image", "classification"),
    ("image", "segmentation"),
]

IMAGE_MODALITY_TASK_MODEL = []
for model in IMAGE_CLASSIFICATION_MODELS:
    IMAGE_MODALITY_TASK_MODEL.append(("image", "classification", model))
for model in IMAGE_SEGMENTATION_MODELS:
    IMAGE_MODALITY_TASK_MODEL.append(("image", "segmentation", model))


def generate_random_class_name():
    ret = "".join(
        [RandomWord().word().capitalize() for _ in range(random.choice(range(4, 10)))]
    )
    ret = ret.replace("-", "").replace(" ", "")
    return ret


# IMAGE_SUBMODELS = [
#     (modality, task, exec(
#         f"type({generate_random_class_name()}, (object,), modlee.model.from_modality_task(modality, task, **kwargs))"))
#     for modality, task, kwargs in IMAGE_MODALITY_TASK_KWARGS
# ]
IMAGE_SUBMODELS = []
for modality, task, kwargs in IMAGE_MODALITY_TASK_KWARGS:
    _var_name = generate_random_class_name()
    _base_modality_task_class = f"{modality.capitalize()}{task.capitalize()}ModleeModel"
    _model = modlee.model.from_modality_task(modality, task, **kwargs)
    _var_name = _var_name.replace("'", "\\'")
    exec(f"class {_var_name}(modlee.model.{_base_modality_task_class}): pass")
    # IMAGE_SUBMODELS.append((
    #     modality, task,
    #     exec(f"{_var_name}(**kwargs)")
    # ))
    exec(f"IMAGE_SUBMODELS.append((modality, task, {_var_name}(**kwargs)))")

""" 
Configure pytest.
"""
import pytest
import inspect
from torchvision import datasets as tv_datasets
from torchvision import models as tvm
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import models as tvm
from torchtext import models as ttm


IMAGE_MODELS = [
    # tvm.resnet18(weights="DEFAULT"),
    # tvm.resnet18(),
    # tvm.resnet50(),
    # tvm.resnet152(),
    tvm.alexnet(),
    # tvm.googlenet(),
]

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
    if 'forward' in dir(tvm_attr_ret):
        print(f"Adding {tvm_attr}")
        IMAGE_MODELS.append(tvm_attr_ret)
        
breakpoint()

IMAGE_SEGMENTATION_MODELS = [
    tvm.segmentation.fcn_resnet50(),
    tvm.segmentation.fcn_resnet101(),
    # tvm.segmentation.lraspp(),
    tvm.segmentation.lraspp_mobilenet_v3_large(),
    tvm.segmentation.deeplabv3_resnet50(),
    tvm.segmentation.deeplabv3_resnet101(),
]
TEXT_MODELS = [
    # ttm.FLAN_T5_BASE,
    # ttm.FLAN_T5_BASE_ENCODER,
    # ttm.FLAN_T5_BASE_GENERATION,
    ttm.ROBERTA_BASE_ENCODER,
    ttm.ROBERTA_DISTILLED_ENCODER,
    # ttm.T5_BASE,
    # ttm.T5_BASE_ENCODER,
    # ttm.T5_SMALL,
    # ttm.T5_SMALL_ENCODER,
    # ttm.T5_SMALL_GENERATION,
    ttm.XLMR_BASE_ENCODER,
    # ttm.XLMR_LARGE_ENCODER, # Too large for M2 MacBook Air?
]


@pytest.fixture()
def dataloaders(batch_size=64):
    training_loader = DataLoader(
        tv_datasets.CIFAR10(
            root="data", train=True, download=True, transform=ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        tv_datasets.CIFAR10(
            root="data", train=False, download=True, transform=ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return training_loader, test_loader

def _check_has_metafeatures(mf, metafeature_types): 

    features = {}
    for metafeature_type in metafeature_types:
        assert hasattr(mf, metafeature_type), f"{mf} has no attribute {metafeature_type}"
        assert isinstance(getattr(mf, metafeature_type), dict), f"{mf} {metafeature_type} is not dictionary"
        # Assert that the attribute is a flat dictionary
        assert not any([isinstance(v,dict) for v in getattr(mf, metafeature_type).values()]), f"{mf} {metafeature_type} not a flat dictionary"
        features.update(getattr(mf, metafeature_type))

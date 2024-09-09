""" 
Configure pytest.
"""

import torch
import pandas as pd
import pytest
import os
import inspect
import modlee
from torchvision import datasets as tv_datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import models as tvm
from torchtext import models as ttm
from pytorch_forecasting import models as pfm

from pytorch_forecasting import NBeats, AutoRegressiveBaseModel
from modlee.timeseries_dataloader import TimeseriesDataset

from .configs import *


def NbeatsInit():
    data = pd.read_csv("data/HDFCBANK.csv")
    data.drop(
        columns=["Series", "Symbol", "Trades", "Deliverable Volume", "Deliverble"],
        inplace=True,
    )
    encoder_column = data.columns.tolist()
    dataset = TimeseriesDataset(
        data=data,
        target="Close",
        time_column="Date",
        encoder_column=encoder_column,
        input_seq=2,
        output_seq=1,
    )
    model = NBeats.from_dataset(dataset=dataset.get_dataset())
    return model
import torch.nn as nn
import torch
import pandas as pd
from omegaconf import OmegaConf

from pytorch_forecasting import models as pfm

def makeDataloader():
    data = pd.read_csv("data/HDFCBANK.csv")
    data.drop(
        columns=["Series", "Symbol", "Trades", "Deliverable Volume", "Deliverble"],
        inplace=True,
    )
    encoder_column = data.columns.tolist()
    dataset = TimeseriesDataset(
        data=data,
        target="Close",
        time_column="Date",
        encoder_column=encoder_column,
        input_seq=2,
        output_seq=1,
    ).to_dataloader(batch_size=1)

    return dataset

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
        assert hasattr(
            mf, metafeature_type
        ), f"{mf} has no attribute {metafeature_type}"
        assert isinstance(
            getattr(mf, metafeature_type), dict
        ), f"{mf} {metafeature_type} is not dictionary"
        # Assert that the attribute is a flat dictionary
        assert not any(
            [isinstance(v, dict) for v in getattr(mf, metafeature_type).values()]
        ), f"{mf} {metafeature_type} not a flat dictionary"
        features.update(getattr(mf, metafeature_type))


# def _check_has_metafeatures_tab(mf, metafeature_types):

def _check_has_metafeatures_tab(mf, metafeature_types):
    # TODO - refactor against the above original function
    features = {}
    for metafeature_type in metafeature_types:
        assert hasattr(mf, metafeature_type), f"{mf} has no attribute {metafeature_type}"
        assert isinstance(getattr(mf, metafeature_type), dict), f"{mf} {metafeature_type} is not dictionary"
        assert not any([isinstance(v,dict) for v in getattr(mf, metafeature_type).values()]), f"{mf} {metafeature_type} not a flat dictionary"
        features.update(getattr(mf, metafeature_type))


def _check_statistical_metafeatures(mf):
    """Check if statistical metafeatures are present in the TabularDataMetafeatures."""
    assert hasattr(
        mf, "features"
    ), "TabularDataMetafeatures does not have 'features' attribute"
    statistical_metafeatures = ["mean", "std_dev", "median", "min", "max", "range"]

    features = getattr(mf, "features")
    for feature in statistical_metafeatures:
        feature_found = any(feature in key for key in features)
        assert feature_found, f"Statistical metafeature '{feature}' is missing"

    assert not any(
        [isinstance(v, dict) for v in features.values()]
    ), "The 'features' dictionary is not flat"


def _check_metafeatures_timesseries(mf, metafeature_types):
    for metafeature_type in metafeature_types:
        assert metafeature_type in mf, f"{mf} has no key {metafeature_type}"


def _model_from_args(modality_task_kwargs):
    modality, task, kwargs = modality_task_kwargs
    return modlee.model.from_modality_task(modality=modality, task=task, **kwargs)


def model_from_args(modality, task, kwargs):
    # modality, task, kwargs = modality_task_kwargs
    return modlee.model.from_modality_task(modality=modality, task=task, **kwargs)


@pytest.fixture(scope="function")
def model(request):
    # breakpoint()
    return _from_modality_task(request, module="model")


@pytest.fixture(scope="function")
def recommender(request):
    return _from_modality_task(request, module="recommender")


def _from_modality_task(request, module="model"):
    modality, task = request.param[:-1]
    return getattr(modlee, module).from_modality_task(
        modality=modality, task=task, **request.param[-1]
    )
def _check_metafeatures_timesseries(mf, metafeature_types):
    for metafeature_type in metafeature_types:
        assert metafeature_type in mf, f"{mf} has no key {metafeature_type}"

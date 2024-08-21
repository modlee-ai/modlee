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
from modlee.model.tabular_model import TabNetModleeModel, CategoryEmbeddingModleeModel, GANDALFModleeModel
from modlee.model.tabular_model import TabTransformerModleeModel, MLP, SNN, EnhancedTabNet, EnhancedResNet

tabnet_config = OmegaConf.create({
    'task': 'classification',
    'embedding_dims': [(3, 2), (5, 3)],
    'metrics_params': [{'task': 'multiclass'}],
    'metrics': ['accuracy'],
    'grouped_features': [
        ['category1', 'category2'],
        ['feature1', 'feature2']
    ],
    'categorical_cols': ['category1', 'category2'],
    'continuous_cols': ['feature1', 'feature2'],
    'continuous_dim': 2,
    'categorical_dim': 2,
    'n_d': 16,
    'n_a': 16,
    'n_steps': 5,
    'gamma': 1.5,
    'n_independent': 2,
    'n_shared': 2,
    'virtual_batch_size': 64,
    'mask_type': 'entmax',
    'embedding_dropout': 0.0,
    'batch_norm_continuous_input': True,
    'learning_rate': 1e-3,
    'loss': 'CrossEntropyLoss',
    'head': 'LinearHead',
        'head_config': {
        'layers': '128-64-32',
        'activation': 'ReLU',
        'use_batch_norm': True,
        'initialization': 'kaiming',
        'dropout': 0.0
    },
    'metrics_prob_input': [False],
    'target_range': []
})

category_embedding_config = OmegaConf.create(
    {
        "task": "classification",
        "head": "LinearHead",
        "layers": "128-64-32",
        "activation": "ReLU",
        "use_batch_norm": True,
        "initialization": "kaiming",
        "dropout": 0.0,
        "head_config": {
            "layers": "128-64-32",
            "activation": "ReLU",
            "use_batch_norm": True,
            "initialization": "kaiming",
            "dropout": 0.0,
        },
        "embedding_dims": [(3, 2), (5, 3)],
        "embedding_dropout": 0.1,
        "batch_norm_continuous_input": True,
        "learning_rate": 1e-3,
        "loss": "CrossEntropyLoss",
        "metrics": ["accuracy"],
        "metrics_params": [{"task": "multiclass"}],
        "metrics_prob_input": [False],
        "target_range": [],
        "categorical_cols": ["feature1", "feature2"],
        "continuous_cols": ["category1", "category2"],
        "continuous_dim": 2,
        "categorical_dim": 2,
        "embedded_cat_dim": 5,
        "virtual_batch_size": 32,
        "seed": 42,
    }
)

gandalf_config = OmegaConf.create({
    'task': 'classification',
    'gflu_stages': 2,
    'gflu_dropout': 0.1,
    'gflu_feature_init_sparsity': 0.3,
    'learnable_sparsity': True,
    'batch_norm_continuous_input': True,
    'embedding_dims': [(3, 2), (5, 3)], 
    'embedding_dropout': 0.1,
    'virtual_batch_size': 32,
    'continuous_dim': 2,  
    'output_dim': 10,  
    'metrics': ['accuracy'],
    'metrics_params': [{'task': 'multiclass'}],
    'metrics_prob_input': [False],
    'seed': 42,
    'head': 'LinearHead',  
    'head_config': {
        'layers': '128-64-32',  
        'activation': 'ReLU',
        'use_batch_norm': True,
        'initialization': 'kaiming',
        'dropout': 0.0
    },
    'loss': 'CrossEntropyLoss'
})

tab_transformer_config = OmegaConf.create({
    'input_embed_dim': 32,  
    'embedding_initialization': 'kaiming_uniform',  
    'embedding_bias': False, 
    'share_embedding': False,  
    'share_embedding_strategy': 'fraction',  
    'shared_embedding_fraction': 0.25, 
    'num_heads': 8,  
    'num_attn_blocks': 6, 
    'transformer_head_dim': 32,  
    'attn_dropout': 0.1,  
    'add_norm_dropout': 0.1, 
    'ff_dropout': 0.1,  
    'ff_hidden_multiplier': 4, 
    'transformer_activation': 'GEGLU', 
    'task': 'classification', 
    'head': 'LinearHead', 
    'head_config': {
        'layers': '128-64-32', 
        'activation': 'ReLU',
        'use_batch_norm': True,
        'initialization': 'kaiming',
        'dropout': 0.0
    },
    'embedding_dims': [(3, 2), (5, 3)], 
    'embedding_dropout': 0.0,  
    'batch_norm_continuous_input': True,  
    'learning_rate': 1e-3,  
    'loss': 'CrossEntropyLoss', 
    'metrics': ['accuracy'],  
    'metrics_params': [{'task': 'multiclass'}],  
    'metrics_prob_input': [False],  
    'target_range': [],  
    'seed': 42,
    'continuous_dim': 2,
    'virtual_batch_size': 32,
    'categorical_dim': 2,
    'categorical_cardinality': [3, 3],
    'output_dim': 10
})

inferred_config = OmegaConf.create({"output_dim": 10})

tabnet_instance = TabNetModleeModel(config=tabnet_config, inferred_config=inferred_config)
category_embedding_instance = CategoryEmbeddingModleeModel(config=category_embedding_config, inferred_config=inferred_config)
gandalf_instance = GANDALFModleeModel(config=gandalf_config, inferred_config=inferred_config)
tab_transformer_instance = TabTransformerModleeModel(config=tab_transformer_config, inferred_config=inferred_config)

input_dim = 10
mlp_model = MLP(input_dim)
snn_model = SNN(input_dim)
tabnet_model = EnhancedTabNet(input_dim)
resnet_model = EnhancedResNet(input_dim)

TABULAR_MODELS = [

    tabnet_instance.get_model(),
    tab_transformer_instance.get_model(),
    category_embedding_instance.get_model(),
    gandalf_instance.get_model()
    
]

CUSTOM_TABULAR_MODELS = [
    mlp_model,
    snn_model,
    tabnet_model,
    resnet_model
]
from pytorch_forecasting import NBeats, AutoRegressiveBaseModel
from modlee.timeseries_dataloader import TimeSeriesDataset

from .configs import *


def NbeatsInit():
    data = pd.read_csv("data/HDFCBANK.csv")
    data.drop(
        columns=["Series", "Symbol", "Trades", "Deliverable Volume", "Deliverble"],
        inplace=True,
    )
    encoder_column = data.columns.tolist()
    dataset = TimeSeriesDataset(
        data=data,
        target="Close",
        time_column="Date",
        encoder_column=encoder_column,
        input_seq=2,
        output_seq=1,
    )
    model = NBeats.from_dataset(dataset=dataset.get_dataset())
    return model

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

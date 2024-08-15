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
from pytorch_tabular.models.tabnet import TabNetModelConfig
from pytorch_tabular.models.tab_transformer import TabTransformerConfig
from pytorch_tabular.models.category_embedding import CategoryEmbeddingModelConfig, CategoryEmbeddingBackbone
from pytorch_tabular.models.danet import DANetConfig
from pytorch_tabular.models.gandalf import GANDALFConfig
from pytorch_tabular import TabularModel
from pytorch_tabular.models.tabnet import TabNetModel
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
import torch.nn as nn
import torch
import pandas as pd
from pytorch_tabular.models.tabnet import TabNetModelConfig, TabNetModel
from omegaconf import OmegaConf
from omegaconf import OmegaConf
from pytorch_tabular.models.category_embedding import CategoryEmbeddingModel
from modlee.model.tabular_model import TabNetModleeModel, CategoryEmbeddingModleeModel

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
    'n_independent': 3,
    'n_shared': 3,
    'virtual_batch_size': 64,
    'mask_type': 'entmax',
    'embedding_dropout': 0.1,
    'batch_norm_continuous_input': True,
    'learning_rate': 1e-3,
    'loss': 'CrossEntropyLoss'
})

category_embedding_config = OmegaConf.create({
    'task': 'classification',
    'head': 'LinearHead',
    'layers': '128-64-32',
    'activation': 'ReLU',
    'use_batch_norm': True,
    'initialization': 'kaiming',
    'dropout': 0.0,
    'head_config': {
        'layers': '128-64-32',
        'activation': 'ReLU',
        'use_batch_norm': True,
        'initialization': 'kaiming',
        'dropout': 0.0
    },
    'embedding_dims': [(10, 5)],  
    'embedding_dropout': 0.1,
    'batch_norm_continuous_input': True,
    'learning_rate': 1e-3,
    'loss': 'CrossEntropyLoss',
    'metrics': ['accuracy'],
    'metrics_params': [{'task': 'multiclass'}],
    'metrics_prob_input': [False],
    'target_range': [],
    'categorical_cols': ['feature1'], 
    'continuous_cols': [], 
    'continuous_dim': 0,
    'categorical_dim': 10,
    'embedded_cat_dim': 5,  
    'virtual_batch_size': 32,
    'seed': 42
})

inferred_config = OmegaConf.create({
    "output_dim": 10
})

tabnet_instance = TabNetModleeModel(config=tabnet_config, inferred_config=inferred_config)
category_embedding_instance = CategoryEmbeddingModleeModel(config=category_embedding_config, inferred_config=inferred_config)
TABULAR_MODELS = [tabnet_instance.get_model(), category_embedding_instance.get_model()]

IMAGE_MODELS = [
    tvm.resnet18(weights="DEFAULT"),
    tvm.resnet18(),
    tvm.resnet50(),
    tvm.resnet152(),
    tvm.googlenet()
]

IMAGE_SEGMENTATION_MODELS = [
    tvm.segmentation.fcn_resnet50(),
    tvm.segmentation.fcn_resnet101(),
    tvm.segmentation.lraspp_mobilenet_v3_large(),
    tvm.segmentation.deeplabv3_resnet50(),
    tvm.segmentation.deeplabv3_resnet101()
]
TEXT_MODELS = [
    ttm.ROBERTA_BASE_ENCODER,
    ttm.ROBERTA_DISTILLED_ENCODER,
    ttm.XLMR_BASE_ENCODER
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

def _check_has_metafeatures_tab(mf, metafeature_types): 

    features = {}
    for metafeature_type in metafeature_types:
        assert hasattr(mf, metafeature_type), f"{mf} has no attribute {metafeature_type}"
        assert isinstance(getattr(mf, metafeature_type), dict), f"{mf} {metafeature_type} is not dictionary"
        # Assert that the attribute is a flat dictionary
        assert not any([isinstance(v,dict) for v in getattr(mf, metafeature_type).values()]), f"{mf} {metafeature_type} not a flat dictionary"
        features.update(getattr(mf, metafeature_type))

def _check_statistical_metafeatures(mf):
    """Check if statistical metafeatures are present in the TabularDataMetafeatures."""
    assert hasattr(mf, 'features'), "TabularDataMetafeatures does not have 'features' attribute"
    statistical_metafeatures = ['mean', 'std_dev', 'median', 'min', 'max', 'range']
    
    features = getattr(mf, 'features')
    for feature in statistical_metafeatures:
        feature_found = any(feature in key for key in features)
        assert feature_found, f"Statistical metafeature '{feature}' is missing"

    assert not any([isinstance(v, dict) for v in features.values()]), "The 'features' dictionary is not flat"


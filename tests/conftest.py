""" 
Configure pytest.
"""
import torch
import pandas as pd
import pytest
import inspect
import modlee
from torchvision import datasets as tv_datasets
from torchvision import models as tvm
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import models as tvm
from torchtext import models as ttm
from pytorch_forecasting import NBeats, AutoRegressiveBaseModel
from modlee.timeseries_dataloader import TimeSeriesDataset



def NbeatsInit():
    data = pd.read_csv('data/HDFCBANK.csv')
    data.drop(columns=['Series', 'Symbol','Trades', 'Deliverable Volume', 'Deliverble'], inplace=True)
    encoder_column = data.columns.tolist()
    dataset = TimeSeriesDataset(data=data, target = 'Close', time_column='Date',
                                                       encoder_column=encoder_column, input_seq=2,
                                                       output_seq=1)
    model = NBeats.from_dataset(
        dataset=dataset.get_dataset()
    )
    return model


def makeDataloader():
    data = pd.read_csv('data/HDFCBANK.csv')
    data.drop(columns=['Series', 'Symbol','Trades', 'Deliverable Volume', 'Deliverble'], inplace=True)
    encoder_column = data.columns.tolist()
    dataset = TimeSeriesDataset(data=data, target = 'Close', time_column='Date',
                                                       encoder_column=encoder_column, input_seq=2,
                                                       output_seq=1).to_dataloader(batch_size=1)
    
    return dataset
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
from modlee.model.tabular_model import TabNetModleeModel, CategoryEmbeddingModleeModel, GANDALFModleeModel
from modlee.model.tabular_model import DANetModleeModel, TabTransformerModleeModel
from pytorch_tabular.models.gandalf import GANDALFModel
from pytorch_tabular.models.danet import DANetModel
from pytorch_forecasting import models as pfm


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
    'embedding_dims': [(3, 2), (5, 3)],  
    'embedding_dropout': 0.1,
    'batch_norm_continuous_input': True,
    'learning_rate': 1e-3,
    'loss': 'CrossEntropyLoss',
    'metrics': ['accuracy'],
    'metrics_params': [{'task': 'multiclass'}],
    'metrics_prob_input': [False],
    'target_range': [],
    'categorical_cols': ['feature1', 'feature2'], 
    'continuous_cols': ['category1', 'category2'], 
    'continuous_dim': 2,
    'categorical_dim': 2,
    'embedded_cat_dim': 5,  
    'virtual_batch_size': 32,
    'seed': 42
})

gandalf_config = OmegaConf.create({
    'task': 'classification',
    'gflu_stages': 2,
    'gflu_dropout': 0.1,
    'gflu_feature_init_sparsity': 0.3,
    'learnable_sparsity': True,
    'batch_norm_continuous_input': True,
    'embedding_dims': [(3, 2), (5, 3)],  # (category_size, embedding_dim)
    'embedding_dropout': 0.1,
    'virtual_batch_size': 32,
    'continuous_dim': 2,  # Number of continuous features
    'output_dim': 10,  # Number of classes for classification
    'metrics': ['accuracy'],
    'metrics_params': [{'task': 'multiclass'}],
    'metrics_prob_input': [False],
    'seed': 42,
    'head': 'LinearHead',  # Specify the head type
    'head_config': {
        'layers': '128-64-32',  # Example architecture for the head
        'activation': 'ReLU',
        'use_batch_norm': True,
        'initialization': 'kaiming',
        'dropout': 0.0
    },
    'loss': 'CrossEntropyLoss'
})

danet_config = OmegaConf.create({
    'task': 'classification',
    'n_layers': 3,
    'abstlay_dim_1': 64,
    'abstlay_dim_2': 32,
    'k': 4,
    'dropout_rate': 0.1,
    'block_activation': 'ReLU',  # Specify the activation function for blocks
    'virtual_batch_size': 64,
    'embedding_dropout': 0.1,
    'batch_norm_continuous_input': True,
    'embedding_dims': [(3, 2), (5, 3)],  # (category_size, embedding_dim)
    'continuous_dim': 2,  # Number of continuous features
    'metrics': ['accuracy'],
    'metrics_params': [{'task': 'multiclass'}],
    'loss': 'CrossEntropyLoss',
    'batch_size': 64,
    'head': 'None',
    'categorical_dim': 2,
    'head': 'LinearHead',
    'head_config': {
        'layers': '128-64-32',
        'activation': 'ReLU',
        'use_batch_norm': True,
        'initialization': 'kaiming',
        'dropout': 0.0}
})

tab_transformer_config = OmegaConf.create({
    'input_embed_dim': 32,  # Embedding dimension for input categorical features
    'embedding_initialization': 'kaiming_uniform',  # Updated Initialization scheme for embedding layers
    'embedding_bias': False,  # Flag to turn on embedding bias
    'share_embedding': False,  # Flag for shared embeddings
    'share_embedding_strategy': 'fraction',  # Strategy for adding shared embeddings
    'shared_embedding_fraction': 0.25,  # Fraction reserved for shared embedding
    'num_heads': 8,  # Number of heads in the Multi-Headed Attention layer
    'num_attn_blocks': 6,  # Number of layers of stacked Multi-Headed Attention layers
    'transformer_head_dim': None,  # Number of hidden units in Multi-Headed Attention layers
    'attn_dropout': 0.1,  # Dropout after Multi-Headed Attention
    'add_norm_dropout': 0.1,  # Dropout in the AddNorm Layer
    'ff_dropout': 0.1,  # Dropout in the Positionwise FeedForward Network
    'ff_hidden_multiplier': 4,  # Multiplier for FF layer scaling
    'transformer_activation': 'GEGLU',  # Activation type in transformer feed forward layers
    'task': 'classification',  # Specify the problem type
    'head': 'LinearHead',  # Type of head used for the model
    'head_config': {
        'layers': '128-64-32',  # Architecture for the head
        'activation': 'ReLU',
        'use_batch_norm': True,
        'initialization': 'kaiming',
        'dropout': 0.0
    },
    'embedding_dims': [(3, 2), (5, 3)],  # Dimensions of embedding for each categorical column
    'embedding_dropout': 0.0,  # Dropout applied to categorical embedding
    'batch_norm_continuous_input': True,  # Normalize continuous layer by BatchNorm
    'learning_rate': 1e-3,  # Learning rate of the model
    'loss': 'CrossEntropyLoss',  # Loss function for classification
    'metrics': ['accuracy'],  # Metrics to track during training
    'metrics_params': [{'task': 'multiclass'}],  # Parameters for metrics function
    'metrics_prob_input': [False],  # Whether input to metrics function is probability or class
    'target_range': [],  # Range for output variable (ignored for multi-target regression)
    'seed': 42,
    'continuous_dim': 2,
    'virtual_batch_size': 32,
    'categorical_dim': 2,
    'categorical_cardinality': [3, 5],
    'output_dim': 10
})

inferred_config = OmegaConf.create({
    "output_dim": 10
})


tabnet_instance = TabNetModleeModel(config=tabnet_config, inferred_config=inferred_config)
category_embedding_instance = CategoryEmbeddingModleeModel(config=category_embedding_config, inferred_config=inferred_config)
gandalf_instance = GANDALFModleeModel(config=gandalf_config, inferred_config=inferred_config)
danet_instance = DANetModleeModel(config=danet_config, inferred_config=inferred_config)
tab_transformer_instance = TabTransformerModleeModel(config=tab_transformer_config, inferred_config=inferred_config)
TABULAR_MODELS = [ danet_instance.get_model(), tab_transformer_instance.get_model(), tabnet_instance.get_model(), category_embedding_instance.get_model(), gandalf_instance.get_model()]
from pytorch_forecasting import NBeats, AutoRegressiveBaseModel
from modlee.timeseries_dataloader import TimeSeriesDataset



def NbeatsInit():
    data = pd.read_csv('data/HDFCBANK.csv')
    data.drop(columns=['Series', 'Symbol','Trades', 'Deliverable Volume', 'Deliverble'], inplace=True)
    encoder_column = data.columns.tolist()
    dataset = TimeSeriesDataset(data=data, target = 'Close', time_column='Date',
                                                       encoder_column=encoder_column, input_seq=2,
                                                       output_seq=1)
    model = NBeats.from_dataset(
        dataset=dataset.get_dataset()
    )
    return model


def makeDataloader():
    data = pd.read_csv('data/HDFCBANK.csv')
    data.drop(columns=['Series', 'Symbol','Trades', 'Deliverable Volume', 'Deliverble'], inplace=True)
    encoder_column = data.columns.tolist()
    dataset = TimeSeriesDataset(data=data, target = 'Close', time_column='Date',
                                                       encoder_column=encoder_column, input_seq=2,
                                                       output_seq=1).to_dataloader(batch_size=1)
    
    return dataset


IMAGE_MODELS = [
    tvm.resnet18(weights="DEFAULT"),
    tvm.resnet18(),
    tvm.resnet50(),
    tvm.resnet152(),
    # tvm.alexnet(),
    tvm.googlenet(),
]

IMAGE_MODALITY_TASK_KWARGS = [
    ("image", "classification", {"num_classes":10}),
    ("image", "segmentation", {"num_classes":10}),
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

TIMESERIES_MODELS = [
    #pfm.TemporalFusionTransformer(),
    #pfm.LSTM(input_size=1, hidden_size=10),
    #pfm.GRU(input_size=1, hidden_size=10, num_layers=1),
    #pfm.RecurrentNetwork()
    #simpleLSTM(),
    simpleModel(),
    conv1dModel(),
    transformerModel(),
    #mlp(input_size=10, output_size=10, hidden_size=[64, 128, 64])
    #pfm.AutoRegressiveBaseModel(),
    #NbeatsInit(),
]
DATALOADER = [
    get_input_for_simple_model(),
    get_input_for_conv1d_model(),
    makeDataloader(),

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





def _check_metafeatures_timesseries(mf, metafeature_types):
    for metafeature_type in metafeature_types:
        assert metafeature_type in mf, f"{mf} has no key {metafeature_type}"
def model_from_args(modality_task_kwargs):
    modality, task, kwargs = modality_task_kwargs
    return modlee.model.from_modality_task(
        modality=modality,
        task=task, 
        **kwargs
    )

# @pytest.mark.parametrize("i", list(range(1)))
# @pytest.fixture
# def iter_fix(i):
#     return i

@pytest.fixture(scope="function")
def get_i(request):
    return request.param**2

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
        modality=modality, 
        task=task,
        **request.param[-1]
    )

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


def _check_metafeatures_timesseries(mf, metafeature_types):
    for metafeature_type in metafeature_types:
        assert metafeature_type in mf, f"{mf} has no key {metafeature_type}"
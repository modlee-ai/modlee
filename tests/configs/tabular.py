import torch
from torch.utils.data import Dataset, DataLoader

try:
    from pytorch_tabular.models.tabnet import TabNetModelConfig
    from pytorch_tabular.models.tab_transformer import TabTransformerConfig
    from pytorch_tabular.models.category_embedding import (
        CategoryEmbeddingModelConfig,
        CategoryEmbeddingBackbone,
    )
    from pytorch_tabular.models.danet import DANetConfig
    from pytorch_tabular.models.gandalf import GANDALFConfig
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models.tabnet import TabNetModel
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models.tabnet import TabNetModelConfig, TabNetModel
    from pytorch_tabular.models.category_embedding import CategoryEmbeddingModel
    from omegaconf import OmegaConf
except:
    pass
import torch.nn as nn
import torch
import pandas as pd
from modlee.model.tabular_model import (
    TabNetModleeModel,
    CategoryEmbeddingModleeModel,
    GANDALFModleeModel,
    TabTransformerModleeModel,
    DANetModleeModel, 
    MLP,
    SNN,
    EnhancedTabNet,
    EnhancedResNet, 
    TabularClassificationModleeModel,
    MODEL_TYPES
)
from modlee.utils import tabular_loaders


# tabnet_config = OmegaConf.create(
#     {
#         "task": "classification",
#         "embedding_dims": [(3, 2), (5, 3)],
#         "metrics_params": [{"task": "multiclass"}],
#         "metrics": ["accuracy"],
#         "grouped_features": [["category1", "category2"], ["feature1", "feature2"]],
#         "categorical_cols": ["category1", "category2"],
#         "continuous_cols": ["feature1", "feature2"],
#         "continuous_dim": 2,
#         "categorical_dim": 2,
#         "n_d": 16,
#         "n_a": 16,
#         "n_steps": 5,
#         "gamma": 1.5,
#         "n_independent": 3,
#         "n_shared": 3,
#         "virtual_batch_size": 64,
#         "mask_type": "entmax",
#         "embedding_dropout": 0.1,
#         "batch_norm_continuous_input": True,
#         "learning_rate": 1e-3,
#         "loss": "CrossEntropyLoss",
#     }
# )

# category_embedding_config = OmegaConf.create(
#     {
#         "task": "classification",
#         "head": "LinearHead",
#         "layers": "128-64-32",
#         "activation": "ReLU",
#         "use_batch_norm": True,
#         "initialization": "kaiming",
#         "dropout": 0.0,
#         "head_config": {
#             "layers": "128-64-32",
#             "activation": "ReLU",
#             "use_batch_norm": True,
#             "initialization": "kaiming",
#             "dropout": 0.0,
#         },
#         "embedding_dims": [(3, 2), (5, 3)],
#         "embedding_dropout": 0.1,
#         "batch_norm_continuous_input": True,
#         "learning_rate": 1e-3,
#         "loss": "CrossEntropyLoss",
#         "metrics": ["accuracy"],
#         "metrics_params": [{"task": "multiclass"}],
#         "metrics_prob_input": [False],
#         "target_range": [],
#         "categorical_cols": ["feature1", "feature2"],
#         "continuous_cols": ["category1", "category2"],
#         "continuous_dim": 2,
#         "categorical_dim": 2,
#         "embedded_cat_dim": 5,
#         "virtual_batch_size": 32,
#         "seed": 42,
#     }
# )

# gandalf_config = OmegaConf.create(
#     {
#         "task": "classification",
#         "gflu_stages": 2,
#         "gflu_dropout": 0.1,
#         "gflu_feature_init_sparsity": 0.3,
#         "learnable_sparsity": True,
#         "batch_norm_continuous_input": True,
#         "embedding_dims": [(3, 2), (5, 3)],  # (category_size, embedding_dim)
#         "embedding_dropout": 0.1,
#         "virtual_batch_size": 32,
#         "continuous_dim": 2,  # Number of continuous features
#         "output_dim": 10,  # Number of classes for classification
#         "metrics": ["accuracy"],
#         "metrics_params": [{"task": "multiclass"}],
#         "metrics_prob_input": [False],
#         "seed": 42,
#         "head": "LinearHead",  # Specify the head type
#         "head_config": {
#             "layers": "128-64-32",  # Example architecture for the head
#             "activation": "ReLU",
#             "use_batch_norm": True,
#             "initialization": "kaiming",
#             "dropout": 0.0,
#         },
#         "loss": "CrossEntropyLoss",
#     }
# )

# danet_config = OmegaConf.create(
#     {
#         "task": "classification",
#         "n_layers": 3,
#         "abstlay_dim_1": 64,
#         "abstlay_dim_2": 32,
#         "k": 4,
#         "dropout_rate": 0.1,
#         "block_activation": "ReLU",  # Specify the activation function for blocks
#         "virtual_batch_size": 64,
#         "embedding_dropout": 0.1,
#         "batch_norm_continuous_input": True,
#         "embedding_dims": [(3, 2), (5, 3)],  # (category_size, embedding_dim)
#         "continuous_dim": 2,  # Number of continuous features
#         "metrics": ["accuracy"],
#         "metrics_params": [{"task": "multiclass"}],
#         "loss": "CrossEntropyLoss",
#         "batch_size": 64,
#         "head": "None",
#         "categorical_dim": 2,
#         "head": "LinearHead",
#         "head_config": {
#             "layers": "128-64-32",
#             "activation": "ReLU",
#             "use_batch_norm": True,
#             "initialization": "kaiming",
#             "dropout": 0.0,
#         },
#     }
# )

# tab_transformer_config = OmegaConf.create(
#     {
#         "input_embed_dim": 32,  # Embedding dimension for input categorical features
#         "embedding_initialization": "kaiming_uniform",  # Updated Initialization scheme for embedding layers
#         "embedding_bias": False,  # Flag to turn on embedding bias
#         "share_embedding": False,  # Flag for shared embeddings
#         "share_embedding_strategy": "fraction",  # Strategy for adding shared embeddings
#         "shared_embedding_fraction": 0.25,  # Fraction reserved for shared embedding
#         "num_heads": 8,  # Number of heads in the Multi-Headed Attention layer
#         "num_attn_blocks": 6,  # Number of layers of stacked Multi-Headed Attention layers
#         "transformer_head_dim": None,  # Number of hidden units in Multi-Headed Attention layers
#         "attn_dropout": 0.1,  # Dropout after Multi-Headed Attention
#         "add_norm_dropout": 0.1,  # Dropout in the AddNorm Layer
#         "ff_dropout": 0.1,  # Dropout in the Positionwise FeedForward Network
#         "ff_hidden_multiplier": 4,  # Multiplier for FF layer scaling
#         "transformer_activation": "GEGLU",  # Activation type in transformer feed forward layers
#         "task": "classification",  # Specify the problem type
#         "head": "LinearHead",  # Type of head used for the model
#         "head_config": {
#             "layers": "128-64-32",  # Architecture for the head
#             "activation": "ReLU",
#             "use_batch_norm": True,
#             "initialization": "kaiming",
#             "dropout": 0.0,
#         },
#         "embedding_dims": [
#             (3, 2),
#             (5, 3),
#         ],  # Dimensions of embedding for each categorical column
#         "embedding_dropout": 0.0,  # Dropout applied to categorical embedding
#         "batch_norm_continuous_input": True,  # Normalize continuous layer by BatchNorm
#         "learning_rate": 1e-3,  # Learning rate of the model
#         "loss": "CrossEntropyLoss",  # Loss function for classification
#         "metrics": ["accuracy"],  # Metrics to track during training
#         "metrics_params": [{"task": "multiclass"}],  # Parameters for metrics function
#         "metrics_prob_input": [
#             False
#         ],  # Whether input to metrics function is probability or class
#         "target_range": [],  # Range for output variable (ignored for multi-target regression)
#         "seed": 42,
#         "continuous_dim": 2,
#         "virtual_batch_size": 32,
#         "categorical_dim": 2,
#         "categorical_cardinality": [3, 5],
#         "output_dim": 10,
#     }
# )

# inferred_config = OmegaConf.create({"output_dim": 10})


# tabnet_instance = TabNetModleeModel(
#     config=tabnet_config, inferred_config=inferred_config
# )
# category_embedding_instance = CategoryEmbeddingModleeModel(
#     config=category_embedding_config, inferred_config=inferred_config
# )
# gandalf_instance = GANDALFModleeModel(
#     config=gandalf_config, inferred_config=inferred_config
# )
# danet_instance = DANetModleeModel(config=danet_config, inferred_config=inferred_config)
# tab_transformer_instance = TabTransformerModleeModel(
#     config=tab_transformer_config, inferred_config=inferred_config
# )
# TABULAR_MODELS = [
#     danet_instance.get_model(),
#     tab_transformer_instance.get_model(),
#     tabnet_instance.get_model(),
#     category_embedding_instance.get_model(),
#     gandalf_instance.get_model(),
# ]
input_dim = 10
mlp_model = MLP(input_dim)
snn_model = SNN(input_dim)
tabnet_model = EnhancedTabNet(input_dim)
resnet_model = EnhancedResNet(input_dim)


TABULAR_MODELS = [
    TabularClassificationModleeModel(
        _type=_type, 
        input_dim=input_dim
    ) for _type in MODEL_TYPES.keys()
]
TABULAR_MODALITY_TASK_KWARGS = [
    ("tabular", "classification", {"_type":"MLP"})
]
TABULAR_MODALITY_TASK_MODEL = [
    ("tabular", "classification", model) for model in TABULAR_MODELS
]
TABULAR_SUBMODELS = TABULAR_MODALITY_TASK_MODEL


# TABULAR_LOADERS = {
#     "housing_dataset": get_housing_dataloader,
#     "adult_dataset": get_adult_dataloader,
#     "diabetes_dataset": get_diabetes_dataloader,
# }
# tabular_loaders = list(TABULAR_LOADERS.values())


import pytest
from . import conftest
import os, re

import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import modlee
from modlee import data_metafeatures as dmf
from modlee.utils import image_loaders, text_loaders
from sklearn.preprocessing import StandardScaler
import spacy


def get_tabular_dataloader(batch_size=32, shuffle=True):
    df = pd.read_csv('housing.csv')
    X = df.drop('MEDV', axis=1).values
    y = df['MEDV'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Create a TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Return a DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


DATA_ROOT = os.path.expanduser("~/efs/.data")
IMAGE_DATALOADER = modlee.utils.get_imagenette_dataloader()
IMAGE_LOADERS = {loader_fn:getattr(image_loaders, loader_fn) for loader_fn in sorted(dir(image_loaders)) if re.match('get_(.*)_dataloader', loader_fn)}
TEXT_LOADERS = {loader_fn:getattr(text_loaders, loader_fn) for loader_fn in dir(text_loaders) if re.match('get_(.*)_dataloader', loader_fn)}
TABULAR_LOADERS = {'get_tabular_dataloader': get_tabular_dataloader}

print('\n'.join(f"image loader{i}: {image_loader}" for i, image_loader in enumerate(IMAGE_LOADERS)))
import pandas as pd 
df = None

@pytest.mark.experimental
class TestDataMetafeatures:
    
    @pytest.mark.parametrize('get_dataloader_fn', IMAGE_LOADERS.values())
    def test_image_dataloader(self, get_dataloader_fn):
        image_mf = dmf.ImageDataMetafeatures(
            get_dataloader_fn(root=DATA_ROOT), testing=True)
        self._check_has_metafeatures(image_mf)

    @pytest.mark.parametrize('get_dataloader_fn', TABULAR_LOADERS.values())
    def test_tabular_dataloader(self, get_dataloader_fn):
        tabular_mf = dmf.TabularDataMetafeatures(
            get_dataloader_fn(root=DATA_ROOT), testing=True)
        self._check_has_metafeatures_tab(tabular_mf)

    @pytest.mark.parametrize('get_dataloader_fn', TEXT_LOADERS.values())
    def test_text_dataloader(self, get_dataloader_fn):
        text_mf = dmf.TextDataMetafeatures(
            get_dataloader_fn(), testing=True)
        self._check_has_metafeatures(text_mf)

    def _check_has_metafeatures_tab(self, mf): 

        metafeature_types = [
            'mfe',
            'properties',
            'features'
        ]
        conftest._check_has_metafeatures(metafeature_types)

    def _check_has_metafeatures(self, mf): 

        metafeature_types = [
            'embedding',
            'mfe',
            'properties'
        ]
        conftest._check_has_metafeatures(mf, metafeature_types)
           
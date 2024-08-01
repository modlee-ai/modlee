import pytest
from . import conftest
import os, re

import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import modlee
from modlee import data_metafeatures as dmf
from modlee.utils import image_loaders, text_loaders, timeseries_loader
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


def get_finance_data(path: str = 'data/HDFCBANK.csv'):
    dataframe = pd.read_csv('data/HDFCBANK.csv')
    ###normalize the dataset
    dataframe.drop(columns=['Symbol', 'Series', 'Trades', 'Deliverable Volume', 'Deliverble'], inplace=True)
    encoder_columns = dataframe.columns.tolist()
    dataloader = timeseries_loader.get_timeseries_dataloader(data=dataframe, input_seq=2, output_seq=1, encoder_column=encoder_columns, target='Close', time_column='Date')
    return dataloader


modlee.init(api_key="GZ4a6OoXmCXUHDJGnnGWNofsPrK0YF0i")
DATA_ROOT = os.path.expanduser("~/efs/.data")
IMAGE_DATALOADER = modlee.utils.get_imagenette_dataloader()
IMAGE_LOADERS = {loader_fn:getattr(image_loaders, loader_fn) for loader_fn in sorted(dir(image_loaders)) if re.match('get_(.*)_dataloader', loader_fn)}
TEXT_LOADERS = {loader_fn:getattr(text_loaders, loader_fn) for loader_fn in dir(text_loaders) if re.match('get_(.*)_dataloader', loader_fn)}
TABULAR_LOADERS = {'get_tabular_dataloader': get_tabular_dataloader}
TIME_SERIES_LOADER = {'finance_data': get_finance_data}
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

    @pytest.mark.parametrize('get_dataloader_fn', TIME_SERIES_LOADER.values())
    def test_timeseries_dataloader(self, get_dataloader_fn):
        timeseries_mf = dmf.TimeSeriesDataMetafeatures(
            get_dataloader_fn()).calculate_metafeatures()
        self._check_has_metafeatures_timeseries(timeseries_mf)

    def _check_has_metafeatures(self, mf): 

        metafeature_types = [
            'embedding',
            'mfe',
            'properties'
        ]
        conftest._check_has_metafeatures(mf, metafeature_types)
        # features = {}
        # for metafeature_type in metafeature_types:
        #     assert hasattr(mf, metafeature_type), f"{mf} has no attribute {metafeature_type}"
        #     assert isinstance(getattr(mf, metafeature_type), dict), f"{mf} {metafeature_type} is not dictionary"
        #     # Assert that the attribute is a flat dictionary
        #     assert not any([isinstance(v,dict) for v in getattr(mf, metafeature_type).values()]), f"{mf} {metafeature_type} not a flat dictionary"
        #     features.update(getattr(mf, metafeature_type))

    def _check_has_metafeatures_timeseries(self, mf):
        metafeature_types = [
            'attr_conc.mean', 'attr_conc.sd', 'attr_ent.mean', 'attr_ent.sd', 
            'cor.mean', 'cor.sd', 'cov.mean', 'cov.sd', 'eigenvalues.mean', 
            'eigenvalues.sd', 'g_mean.mean', 'g_mean.sd', 'h_mean.mean', 
            'h_mean.sd', 'iq_range.mean', 'iq_range.sd', 'kurtosis.mean', 
            'kurtosis.sd', 'mad.mean', 'mad.sd', 'max.mean', 'max.sd', 
            'mean.mean', 'mean.sd', 'median.mean', 'median.sd', 'min.mean', 
            'min.sd', 'nr_cor_attr', 'nr_norm', 'nr_outliers', 'range.mean', 
            'range.sd', 'sd.mean', 'sd.sd', 'skewness.mean', 'skewness.sd', 
            'sparsity.mean', 'sparsity.sd', 't_mean.mean', 't_mean.sd', 
            'var.mean', 'var.sd', 'combined_quantile_25', 'combined_quantile_50', 
            'combined_quantile_75', 'combined_autocorr_lag1', 
            'combined_partial_autocorr_lag1', 'combined_trend_strength', 
            'combined_seasonal_strength'
        ]
        conftest._check_metafeatures_timesseries(mf, metafeature_types)

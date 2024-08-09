import pytest
from . import conftest
import os, re

import torch
from torch.utils.data import DataLoader
import torchvision

import modlee
from modlee import data_metafeatures as dmf
from modlee.utils import image_loaders, text_loaders, timeseries_loader

import spacy

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
# TEXT_DATALOADER = modlee.utils.get_wnli_dataloader() 

IMAGE_LOADERS = {loader_fn:getattr(image_loaders, loader_fn) for loader_fn in sorted(dir(image_loaders)) if re.match('get_(.*)_dataloader', loader_fn)}
TEXT_LOADERS = {loader_fn:getattr(text_loaders, loader_fn) for loader_fn in dir(text_loaders) if re.match('get_(.*)_dataloader', loader_fn)}
TIME_SERIES_LOADER = {'finance_data': get_finance_data}
print('\n'.join(f"image loader{i}: {image_loader}" for i, image_loader in enumerate(IMAGE_LOADERS)))
import pandas as pd 
# df = pd.DataFrame()
df = None

@pytest.mark.experimental
class TestDataMetafeatures:
    
    @pytest.mark.parametrize('get_dataloader_fn', IMAGE_LOADERS.values())
    def test_image_dataloader(self, get_dataloader_fn):
        image_mf = dmf.ImageDataMetafeatures(
            get_dataloader_fn(root=DATA_ROOT), testing=True)
        self._check_has_metafeatures(image_mf)

    def test_tabular_dataloader(self, get_dataloader_fn):
        tabular_mf = dmf.TabularDataMetafeatures(
            get_dataloader_fn(), testing=True
        )

    # @pytest.mark.parametrize('get_dataloader_fn', [
    #     modlee.utils.get_mnli_dataloader,
    #     modlee.utils.get_cola_dataloader,
    #     modlee.utils.get_wnli_dataloader,
    #     ])
    # @pytest.mark.parametrize('get_dataloader_fn', [modlee.utils.get_wnli_dataloader])
    @pytest.mark.parametrize('get_dataloader_fn', TEXT_LOADERS.values())
    def test_text_dataloader(self, get_dataloader_fn):
        text_mf = dmf.TextDataMetafeatures(
            get_dataloader_fn(root=DATA_ROOT), testing=True)
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
        self._check_metafeatures(mf, metafeature_types)

    def _check_metafeatures(self, mf, metafeature_types):
        for metafeature_type in metafeature_types:
            assert metafeature_type in mf, f"{mf} has no key {metafeature_type}"

           
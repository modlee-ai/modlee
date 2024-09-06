import pytest
from . import conftest
import os, re
import pandas as pd
from functools import partial

import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import modlee
from modlee import data_metafeatures as dmf
# from modlee.utils import image_loaders, text_loaders, timeseries_loader
from sklearn.preprocessing import StandardScaler

import spacy
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
from pytorch_tabular.models import CategoryEmbeddingModelConfig

from .configs import *


# DATA_ROOT = os.path.expanduser("~/efs/.data")
DATA_ROOT = os.path.expanduser("./.data")
# IMAGE_DATALOADER = modlee.utils.get_imagenette_dataloader()

# IMAGE_LOADERS = {
#     loader_fn: getattr(image_loaders, loader_fn)
#     for loader_fn in sorted(dir(image_loaders))
#     if re.match("get_(.*)_dataloader", loader_fn)
# }
# TEXT_LOADERS = {
#     loader_fn: getattr(text_loaders, loader_fn)
#     for loader_fn in dir(text_loaders)
#     if re.match("get_(.*)_dataloader", loader_fn)
# }
# TODO - add timeseries to modalities

"""construct IMAGE_LOADERS, TEXT_LOADERS, etc"""
globals().update(
    {
        f"{modality.upper()}_LOADERS": {
            loader_fn: getattr(globals()[f"{modality}_loaders"], loader_fn)
            for loader_fn in dir(globals()[f"{modality}_loaders"])
            if re.match("get_(.*)_dataloader", loader_fn)
        }
        for modality in conftest.MODALITIES
    }
)
# breakpoint()

# df = pd.DataFrame()
df = None


class TestDataMetafeatures:
    # @pytest.mark.parametrize("get_dataloader_fn", TABULAR_LOADERS.values())
    # def test_tabular_dataloader(self, get_dataloader_fn):
    #     tabular_mf = dmf.TabularDataMetafeatures(get_dataloader_fn(), testing=True)
    #     self._check_has_metafeatures_tab(tabular_mf)
    #     self._check_statistical_metafeatures(tabular_mf)

    # @pytest.mark.parametrize('get_dataloader_fn', TEXT_LOADERS.values())
    # def test_text_dataloader(self, get_dataloader_fn):
    #     text_mf = dmf.TextDataMetafeatures(
    #         get_dataloader_fn(), testing=True)
    #     self._check_has_metafeatures(text_mf)

    def _test_dataloader(self, modality, get_dataloader_fn):
        data_mf = dmf.from_modality_task(
            modality,
            task="",
            dataloader=get_dataloader_fn(root=DATA_ROOT),
            testing=True,
        )
        assert type(data_mf).__name__ == f"{modality.capitalize()}DataMetafeatures"
        self._check_has_metafeatures(data_mf)

    for modality in conftest.MODALITIES:
        _loaders = list(globals()[f"{modality.upper()}_LOADERS"].values())
        _loaders = zip([modality] * len(_loaders), _loaders)

        def _func(self, modality, get_dataloader_fn):
            return self._test_dataloader(modality, get_dataloader_fn=get_dataloader_fn)

        _f = pytest.mark.parametrize("modality, get_dataloader_fn", _loaders)(_func)
        locals().update({f"test_{modality}_dataloader": _f})

    def _check_has_metafeatures_tab(self, mf):
        metafeature_types = ["mfe", "properties", "features"]
        conftest._check_has_metafeatures_tab(mf, metafeature_types)

    @pytest.mark.parametrize("get_dataloader_fn", TIMESERIES_LOADERS.values())
    def test_timeseries_dataloader(self, get_dataloader_fn):
        timeseries_mf = dmf.TimeseriesDataMetafeatures(
            get_dataloader_fn()
        ).calculate_metafeatures()
        self._check_has_metafeatures_timeseries(timeseries_mf)

    def _check_has_metafeatures(self, mf):
        metafeature_types = ["embedding", "mfe", "properties"]
        conftest._check_has_metafeatures(mf, metafeature_types)

    def _check_statistical_metafeatures(self, mf):
        conftest._check_statistical_metafeatures(mf)
        # features = {}
        # for metafeature_type in metafeature_types:
        #     assert hasattr(mf, metafeature_type), f"{mf} has no attribute {metafeature_type}"
        #     assert isinstance(getattr(mf, metafeature_type), dict), f"{mf} {metafeature_type} is not dictionary"
        #     # Assert that the attribute is a flat dictionary
        #     assert not any([isinstance(v,dict) for v in getattr(mf, metafeature_type).values()]), f"{mf} {metafeature_type} not a flat dictionary"
        #     features.update(getattr(mf, metafeature_type))

    def _check_has_metafeatures_timeseries(self, mf):
        metafeature_types = [
            "attr_conc.mean",
            "attr_conc.sd",
            "attr_ent.mean",
            "attr_ent.sd",
            "cor.mean",
            "cor.sd",
            "cov.mean",
            "cov.sd",
            "eigenvalues.mean",
            "eigenvalues.sd",
            "g_mean.mean",
            "g_mean.sd",
            "h_mean.mean",
            "h_mean.sd",
            "iq_range.mean",
            "iq_range.sd",
            "kurtosis.mean",
            "kurtosis.sd",
            "mad.mean",
            "mad.sd",
            "max.mean",
            "max.sd",
            "mean.mean",
            "mean.sd",
            "median.mean",
            "median.sd",
            "min.mean",
            "min.sd",
            "nr_cor_attr",
            "nr_norm",
            "nr_outliers",
            "range.mean",
            "range.sd",
            "sd.mean",
            "sd.sd",
            "skewness.mean",
            "skewness.sd",
            "sparsity.mean",
            "sparsity.sd",
            "t_mean.mean",
            "t_mean.sd",
            "var.mean",
            "var.sd",
            "combined_quantile_25",
            "combined_quantile_50",
            "combined_quantile_75",
            "combined_autocorr_lag1",
            "combined_partial_autocorr_lag1",
            "combined_trend_strength",
            "combined_seasonal_strength",
        ]
        conftest._check_metafeatures_timesseries(mf, metafeature_types)

    _check_has_metafeatures = partial(
        conftest._check_has_metafeatures,
        metafeature_types=["embedding", "mfe", "properties"],
    )

import pytest
import re

import torch
from torch.utils.data import DataLoader
import torchvision

import modlee
from modlee import data_metafeatures as dmf
from modlee.utils import image_loaders, text_loaders

import spacy

IMAGE_DATALOADER = modlee.utils.get_imagenette_dataloader()
# TEXT_DATALOADER = modlee.utils.get_wnli_dataloader() 


IMAGE_LOADERS = {loader_fn:getattr(image_loaders, loader_fn) for loader_fn in dir(image_loaders) if re.match('get_(.*)_dataloader', loader_fn)}
TEXT_LOADERS = {loader_fn:getattr(text_loaders, loader_fn) for loader_fn in dir(text_loaders) if re.match('get_(.*)_dataloader', loader_fn)}

import pandas as pd 
# df = pd.DataFrame()
df = None

class TestDataMetafeatures:
    
    @pytest.mark.parametrize('get_dataloader_fn', IMAGE_LOADERS.values())
    def test_image_dataloader(self, get_dataloader_fn):
        image_mf = dmf.ImageDataMetafeatures(get_dataloader_fn(), testing=True)
        self._check_has_metafeatures(image_mf)

    # @pytest.mark.parametrize('get_dataloader_fn', [
    #     modlee.utils.get_mnli_dataloader,
    #     modlee.utils.get_cola_dataloader,
    #     modlee.utils.get_wnli_dataloader,
    #     ])
    # @pytest.mark.parametrize('get_dataloader_fn', [modlee.utils.get_wnli_dataloader])
    @pytest.mark.parametrize('get_dataloader_fn', TEXT_LOADERS.values())
    def test_text_dataloader(self, get_dataloader_fn):
        text_mf = dmf.TextDataMetafeatures(get_dataloader_fn(), testing=True)
        self._check_has_metafeatures(text_mf)

    def _check_has_metafeatures(self, mf): 
        metafeature_types = [
            'embedding',
            'mfe',
            'properties'
        ]
        features = {}
        for metafeature_type in metafeature_types:
            assert hasattr(mf, metafeature_type), f"{mf} has no attribute {metafeature_type}"
            assert isinstance(getattr(mf, metafeature_type), dict), f"{mf} {metafeature_type} is not dictionary"
            # Assert that the attribute is a flat dictionary
            assert not any([isinstance(v,dict) for v in getattr(mf, metafeature_type).values()]), f"{mf} {metafeature_type} not a flat dictionary"
            features.update(getattr(mf, metafeature_type))
            
        global df
        # breakpoint()
        if df is None:
            df = pd.DataFrame(features)
        else:
            try:
                df = pd.concat([df, pd.DataFrame(features)]).reset_index()
            except:
                pass
            # df = df.append(features, ignore_index=True)
            # df.update(features)
        # breakpoint()
        # breakpoint()
        
    

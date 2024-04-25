# %%
import pytest
import re, os

import torch
from torch.utils.data import DataLoader
import torchvision

import modlee
from modlee import data_metafeatures as dmf
from modlee.utils import text_loaders, image_loaders

import pandas as pd
import spacy

DATA_ROOT = os.path.expanduser("~/efs/.data")
IMAGE_DATALOADER = modlee.utils.get_imagenette_dataloader()
# TEXT_DATALOADER = modlee.utils.get_wnli_dataloader() 


TEXT_LOADERS = {loader_fn:getattr(text_loaders, loader_fn) for loader_fn in dir(text_loaders) if re.match('get_(.*)_dataloader', loader_fn)}
IMAGE_LOADERS = [getattr(image_loaders, loader_fn) for loader_fn in dir(image_loaders) if re.match('get_(.*)_dataloader', loader_fn)]


# %%
mf_global = None
def get_df_from_loaders(loaders, modality, n_samples=1):
    global mf_global
    if isinstance(loaders, dict):
        loaders = list(loaders.values())
    df = pd.DataFrame()
    print(loaders)
    features = []
    MFClass = getattr(dmf, f"{modality.capitalize()}DataMetafeatures")
    for loader_fn in loaders:
        for _ in range(n_samples):
            metafeatures = MFClass(
                loader_fn(root=DATA_ROOT), testing=True
            )
            if hasattr(loader_fn, 'args'):
                dataset_name = loader_fn.args[0]
            else:
                dataset_name = loader_fn.__name__
            mf_global = metafeatures
            features.append({
                    'dataset_name':dataset_name,
                    **metafeatures.embedding,
                    **metafeatures.mfe,
                    **metafeatures.properties,
            })
            pd.DataFrame(features[-1]).to_csv(
                f'./{modality}_features_cache.csv',
                mode='a')
    df = pd.DataFrame(features)
    return df



# %%
text_df = get_df_from_loaders(TEXT_LOADERS, 'text')

# %%
image_df = get_df_from_loaders(IMAGE_LOADERS[18:], 'image', n_samples=4)

# %%
image_df

# %%
mf_dict = {
    # **mf_global.embedding,
    # **mf_global.mfe,
    **mf_global.properties
}
pd.DataFrame(mf_dict,)
print(mf_dict)

# %%
text_df['skewness.mean_0']

# %%
df = pd.DataFrame(features)
# print(len(TEXT_LOADERS))
df = df.fillna(0)

# %%
# print(df.dtypes)
import numpy as np
object_columns = df.select_dtypes(include=['object']).columns
df[object_columns] = df[object_columns].apply(
    lambda x : x[0]
)
df.to_csv('text_metafeatures.tsv', sep='\t', index=False, header=False)

# %%
def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())

# Normalize DataFrame by columns
normalized_df = df.apply(min_max_normalize)
normalized_df.to_csv(
    'text_metafeatures_normalized.tsv', 
    sep='\t', 
    index=False,
    header=False
    )
with open("data_labels.txt",'w') as _file:
    _file.write('\n'.join(labels))
    # _file.write('\n'.join(list(TEXT_LOADERS.keys())))

# %%
embd_cols = sorted(col for col in normalized_df.columns if 'embd' in col)
print(embd_cols)
normalized_df[embd_cols].to_csv(
    'text_metafeatures_normalized_embd.tsv',
    sep='\t',
    index=False,
    header=False
)
normalized_df.drop(columns=embd_cols).to_csv(
    'text_metafeatures_normalized_mfe.tsv',
    sep='\t',
    index=False,
    header=False
)
print(list(TEXT_LOADERS.keys()), sep='\n')



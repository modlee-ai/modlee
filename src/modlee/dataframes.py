import pandas as pd
import numpy as np
from functools import partial

class ModleeDataFrame(pd.DataFrame):
    def to_tsv(self, *args, **kwargs):
        super().to_csv(
            *args, 
            sep="\t",
            index=False,
            header=False,
            **kwargs,
        )

    def to_txt(self, path, columns=None, apply_fn=None, *args, **kwargs):
        df_txt = self
        if columns is not None:
            df_txt = df_txt[columns]
        if apply_fn is not None:
            df_txt = df_txt.apply(apply_fn)
        with open(path, 'w') as _file:
            _file.write('\n'.join(
                list(df_txt)
            ))
        

class DataFrameTransforms:
    @staticmethod
    def list_cols2item(df):
        object_columns = df.select_dtypes(include=['object']).columns
        df[object_columns] = df[object_columns].apply(
            lambda x : np.prod(x)
        )
        return df
    
    @staticmethod
    def drop_nonnum(df):
        return df.select_dtypes(include=['float','int'])
        
    @staticmethod
    def fillna(df, val=0):
        return df.fillna(val)
    
    @staticmethod
    def dropna(df):
        return df.dropna(axis=1, how='any')

    @staticmethod
    def normalize(df):
        def min_max_normalize(column):
            return (column - column.min()) / (column.max() - column.min())
        return df.apply(min_max_normalize)

    @staticmethod
    def compose(transforms):
        def apply_transforms(df):
            for transform in transforms:
                df = transform(df)
            return df
        return apply_transforms
    
default_transforms = DataFrameTransforms.compose([
    DataFrameTransforms.list_cols2item,
    DataFrameTransforms.drop_nonnum,
    DataFrameTransforms.normalize,
    DataFrameTransforms.dropna
])
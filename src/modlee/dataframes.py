import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from functools import partial


class ModleeDataFrame(pd.DataFrame):
    """
    A wrapper class over a pandas DataFrame with convenience functions for metafeatures.
    """

    def to_tsv(self, *args, **kwargs):
        """
        Save a dataframe to a tab-separated value file.
        Wraps around and passes arguments to pd.DataFrame.to_csv().
        """
        super().to_csv(
            *args,
            sep="\t",
            index=False,
            header=False,
            **kwargs,
        )

    def to_txt(self, path, columns=None, apply_fn=None, *args, **kwargs):
        """
        Save a column of a dataframe

        :param path: The path to save the txt to.
        :param columns: The columns of the dataframe to save, defaults to None
        :param apply_fn: An optional function to apply to the dataframe (i.e. pd.DataFrame.apply(apply_fn)), defaults to None
        """

        df_txt = self
        if columns is not None:
            df_txt = df_txt[columns]
        if apply_fn is not None:
            df_txt = df_txt.apply(apply_fn)
        with open(path, "w") as _file:
            _file.write("\n".join(list(df_txt)))


class DataFrameTransforms:
    """
    Transforms for a dataframe.
    Inspired by torchvision's transforms.
    """

    @staticmethod
    def list_cols2item(df):
        object_columns = df.select_dtypes(include=["object"]).columns
        df[object_columns] = df[object_columns].apply(lambda x: np.prod(list(x)))
        return df

    @staticmethod
    def obj2num(df):
        def _series2num(x):
            try:
                pd.to_numeric(x)
                return pd.to_numeric(x)
            except:
                return x

        return df.apply(_series2num)

    @staticmethod
    def drop_nonnum(df):
        return df.select_dtypes(include=["float", "int"])

    @staticmethod
    def fillna(df, val=0):
        return df.fillna(val)

    @staticmethod
    def dropna(df):
        return df.dropna(axis=1, how="any")

    @staticmethod
    def normalize(df):
        def min_max_normalize(column):
            return (column - column.min()) / (column.max() - column.min())

        return df.apply(min_max_normalize)

    @staticmethod
    def scale(df, scaler=None):
        """
        Cannot use this in compose because it also returns the scaler
        """
        columns = df.columns
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(df)
        df = scaler.transform(df)
        df = pd.DataFrame(df, columns=columns)
        return df, scaler

    @staticmethod
    def compose(transforms):
        def apply_transforms(df):
            for transform in transforms:
                df = transform(df)
            return df

        return apply_transforms


DEFAULT_TRANSFORMS = [
    DataFrameTransforms.list_cols2item,
    DataFrameTransforms.drop_nonnum,
    DataFrameTransforms.dropna,
]
default_transforms = DataFrameTransforms.compose(DEFAULT_TRANSFORMS)

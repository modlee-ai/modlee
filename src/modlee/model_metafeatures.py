from abc import abstractmethod
from functools import partial

import numpy as np
import pandas as pd

import torch

import modlee
from modlee.utils import get_model_size, class_from_modality_task, get_modality_task

converter = modlee.converter.Converter()

from modlee.utils import INPUT_DUMMY

class ModelMetafeatures:
    def __init__(self, torch_model: torch.nn.Module, sample_input=None, *args, **kwargs):
        self.torch_model = torch_model
        self.modality, self.task = get_modality_task(torch_model)
        self.sample_input = sample_input

        if sample_input is None:
            sample_input = INPUT_DUMMY[self.modality]
            print("Using default sample input")
        else:
            # Check the input type
            # self.check_input_type(sample_input)

            # Unpack if it's a list of length 1
            if isinstance(sample_input, list) and len(sample_input) == 1:
                sample_input = sample_input[0]
                #print(f"Unpacked sample_input: {type(sample_input)}")

            # Convert to tensor if it's not already a tensor
            if not torch.is_tensor(sample_input):
                try:
                    sample_input = torch.tensor(sample_input)
                    
                except Exception as e:
                    print(f"Error converting sample_input to tensor: {e}")

        self.onnx_graph = converter.torch_model2onnx_graph(
            torch_model, 
            input_dummy=sample_input)

        # Must calculate NetworkX before initializing tensors
        self.onnx_nx = converter.index_nx(converter.onnx_graph2onnx_nx(self.onnx_graph))
        self.onnx_text = converter.onnx_graph2onnx_text(self.onnx_graph)
        self.onnx_graph = converter.init_onnx_tensors(
            converter.init_onnx_params(self.onnx_graph)
        )

        self.dataframe = self.get_graph_dataframe(self.onnx_graph)
        self.properties = self.get_properties()

    def check_input_type(self, sample_input):
        """Prints the type and shape of the sample_input and breaks for debugging."""
        #print(f"Input type: {type(sample_input)}")
        if isinstance(sample_input, np.ndarray):
            print(f"Input shape (NumPy array): {sample_input.shape}")
        elif isinstance(sample_input, list):
            print(f"Input is a list with length: {len(sample_input)}")
        elif torch.is_tensor(sample_input):
            print(f"Input is a tensor with shape: {sample_input.shape}")
        else:
            print("Unknown input type")

    def get_properties(self, *args, **kwargs):
        '''
        These are:
            - Layer counts
            - Layer parameter stats, e.g. min/max/mean conv sizes
            - Size
            - Input / output shapes
        Reference the ModelMFE (metafeature extractor): https://github.com/modlee-ai/recommender/blob/a86eb715c0f8771bbcb20a624eb20bc6f07d6c2b/data_prep/model_mfe.py#L117
        In that prior implementation, used the ONNX text representation, and regexes
        '''
        
        return {
            "size": get_model_size(self.torch_model, as_MB=False),
            "output_shape": self.get_output_shape(),
            **self.get_parameter_statistics(self.dataframe),
            **self.get_layer_counts(self.dataframe),
        }

    @abstractmethod
    def get_output_shape(self):
        output = self.torch_model(self.sample_input)
        return np.array(output.shape[1:])

    @staticmethod
    def get_graph_dataframe(onnx_graph, *args, **kwargs):
        """
        Parse the layers of the model, maybe as a dataframe?
        With columns of layer type, parameters, indices (position in graph)
        Can then calculate parameters e.g. counts, parameter ranges, etc
        This almost seems like a converter function
        """
        nodes = []
        for node_idx, node in enumerate(onnx_graph.nodes):
            node_op = node.op.lower()
            node_dict = {
                "operation": node_op,
                "index": node_idx,
                # Attributes specific to this node operation, e.g. convolution kernel sizes
                **{
                    f"{node_op}_{node_attr_key}": node_attr_val
                    for node_attr_key, node_attr_val in node.attrs.items()
                },
            }

            nodes.append(node_dict)

        df = pd.DataFrame(nodes)
        df = ModelMetafeatures.dataframe_lists_to_columns(df)
        return df

    @staticmethod
    def dataframe_lists_to_columns(df: pd.DataFrame):
        """
        Split dataframe columns that are lists to separate, indexed columns

        :param df: _description_
        """
        object_cols = df.select_dtypes(include="object").columns
        list_cols = [
            col for col in object_cols if isinstance(df[col].dropna().iloc[0], list)
        ]
        for list_col in list_cols:
            # Turn the lists into a dataframe, with number of columns equal to the max length of a list
            list_df = pd.DataFrame(
                df[list_col].apply(lambda x: x if isinstance(x, list) else []).to_list()
            )
            # Get the number of columns created
            n_cols = list_df.shape[1]
            df[[f"{list_col}_{i}" for i in range(n_cols)]] = list_df
        df = df.drop(columns=list_cols)
        return df

    @staticmethod
    def get_layer_counts(df: pd.DataFrame):
        """
        Get the counts of each layer type in a dataframe

        :param df: _description_
        """
        count_dict = dict(df["operation"].value_counts())
        count_dict = {f"{k}_count": v for k, v in count_dict.items()}
        return count_dict

    @staticmethod
    def get_parameter_statistics(df: pd.DataFrame | pd.Series):
        """
        Get the statistics of a single-column dataframe or series

        :param df: _description_
        """
        statistics = ["min", "max", "mean", "median", "std"]
        if isinstance(df, pd.DataFrame):
            df_float = df.select_dtypes(include="float")
        else:
            df_float = pd.DataFrame(df)
        ret = {}
        for col in df_float.columns:
            for statistic in statistics:
                ret.update(
                    {f"{col}_{statistic}": getattr(np, statistic)(df_float[col])}
                )

        return ret

class ImageModelMetafeatures(ModelMetafeatures):
    def get_output_shape(
        self,
    ):
        output = self.torch_model(self.sample_input)
        return np.array(output.shape[1:])

class ImageClassificationModelMetafeatures(ImageModelMetafeatures):
    pass

class ImageSegmentationModelMetafeatures(ImageModelMetafeatures):
    def get_output_shape(self):
        output = self.torch_model(self.sample_input)
        if isinstance(output, dict):
            output = output["out"]
        return np.array(output.shape[1:])

class ImageRegressionModelMetafeatures(ImageModelMetafeatures):
    pass

class ImageImageToImageModelMetafeatures(ImageModelMetafeatures):
    pass

class TabularModelMetafeatures(ModelMetafeatures):
    pass

class TabularClassificationModelMetafeatures(TabularModelMetafeatures):
    pass

class TabularRegressionModelMetafeatures(TabularModelMetafeatures):
    pass

class TimeseriesModelMetafeatures(ModelMetafeatures):
    pass
    
class TimeseriesForecastingModelMetafeatures(TimeseriesModelMetafeatures):
    pass
class TimeseriesClassificationModelMetafeatures(TimeseriesModelMetafeatures):
    pass
class TimeseriesRegressionModelMetafeatures(TimeseriesModelMetafeatures):
    pass
from_modality_task = partial(class_from_modality_task, _class="Model_Metafeatures")


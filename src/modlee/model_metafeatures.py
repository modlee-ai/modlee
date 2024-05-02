from abc import abstractmethod

import numpy as np
import pandas as pd

import torch
import modlee
converter = modlee.converter.Converter()

class ModelMetafeatures:
    @abstractmethod
    def __init__(self, torch_model: torch.nn.Module, *args, **kwargs):
        #  Should work with any of the available model representations
        # Torch model/text, ONNX graph/text
        # Store these different representations
        self.torch_model = torch_model
        self.onnx_graph = converter.torch_model2onnx_graph(
                    self.torch_model
                )
        self.onnx_text = converter.onnx_graph2onnx_text(self.onnx_graph)
        self.onnx_graph = converter.init_onnx_tensors(
            converter.init_onnx_params(
                self.onnx_graph
            )
        )
        
        self.dataframe = self.get_graph_dataframe(self.onnx_graph)
        pass
    
    @abstractmethod
    def get_embedding(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def get_properties(self, *args, **kwargs):
        # These are: 
        # - Layer counts
        # - Layer parameter stats, e.g. min/max/mean conv sizes
        # - Size
        # - Input / output shapes
        # Reference the ModelMFE (metafeature extractor): https://github.com/modlee-ai/recommender/blob/a86eb715c0f8771bbcb20a624eb20bc6f07d6c2b/data_prep/model_mfe.py#L117
        # In that prior implementation, used the ONNX text representation, and regexes
        pass

    def get_size(self, *args, **kwargs):
        """
        Get the size of the model in bytes
        """
        pass
    
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
                'operation': node_op,
                'index': node_idx,
                # Attributes specific to this node operation, e.g. convolution kernel sizes
                **{f'{node_op}_{node_attr_key}':node_attr_val \
                    for node_attr_key, node_attr_val \
                        in node.attrs.items()} 
            }

            nodes.append(node_dict)
            
        df = pd.DataFrame(nodes)
        df = ModelMetafeatures.dataframe_lists_to_columns(df)
        # breakpoint()
        return df
    
    @staticmethod
    def dataframe_lists_to_columns(df: pd.DataFrame):
        """
        Split dataframe columns that are lists to separate, indexed columns

        :param df: _description_
        """
        object_cols = df.select_dtypes(include='object').columns
        list_cols = [col for col in object_cols if isinstance(df[col].dropna().iloc[0], list)]
        for list_col in list_cols:
            # Turn the lists into a dataframe, with number of columns equal to the max length of a list
            list_df = pd.DataFrame(df[list_col].apply(lambda x: x if isinstance(x, list) else []).to_list())
            # Get the number of columns created
            n_cols = list_df.shape[1]
            df[[f"{list_col}_{i}" for i in range(n_cols)]] = list_df
        df = df.drop(columns=list_cols)
        return df
        pass

    @staticmethod
    def get_layer_counts(df: pd.DataFrame):
        """
        Get the counts of each layer type in a dataframe

        :param df: _description_
        """
        count_dict = dict(df['operation'].value_counts())
        count_dict = {f"{k}_count":v for k,v in count_dict.items()}
        return count_dict

    @staticmethod
    def get_parameter_statistics(df: pd.DataFrame | pd.Series):
        """
        Get the statistics of a single-column dataframe or series

        :param df: _description_
        """
        statistics = ['min', 'max', 'mean', 'median', 'std']
        return {
            statistic:getattr(np, statistic)(df) for statistic in statistics
        }
        pass

class ImageModelMetafeatures(ModelMetafeatures):
    pass

class TextModelMetafeatures(ModelMetafeatures):
    pass
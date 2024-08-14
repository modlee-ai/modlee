from abc import abstractmethod
from functools import partial

import numpy as np
import pandas as pd
import karateclub
import pickle

import torch
import modlee
from modlee.config import G2V_PKL
from modlee.utils import get_model_size, class_from_modality_task

converter = modlee.converter.Converter()

# g2v = ModelEncoder.from()


class ModelMetafeatures:
    def __init__(self, torch_model: torch.nn.Module, *args, **kwargs):
        #  Should work with any of the available model representations
        # Torch model/text, ONNX graph/text
        # Store these different representations
        self.torch_model = torch_model
        # self.torch_model.to(device=modlee.DEVICE)
        self.onnx_graph = converter.torch_model2onnx_graph(self.torch_model)
        # Must calculate NetworkX before initializing tensors
        self.onnx_nx = converter.index_nx(converter.onnx_graph2onnx_nx(self.onnx_graph))
        self.onnx_text = converter.onnx_graph2onnx_text(self.onnx_graph)
        self.onnx_graph = converter.init_onnx_tensors(
            converter.init_onnx_params(self.onnx_graph)
        )

        self.dataframe = self.get_graph_dataframe(self.onnx_graph)
        self.properties = self.get_properties()
        self.embedding = self.get_embedding()
        pass

    def get_embedding(self, *args, **kwargs):
        g2v = ModelEncoder.from_pkl(G2V_PKL)
        embd = g2v.infer([self.onnx_nx])[0]
        embd_dict = {f"embd_{i}": e for i, e in enumerate(embd)}
        return embd_dict

    def get_properties(self, *args, **kwargs):
        # These are:
        # - Layer counts
        # - Layer parameter stats, e.g. min/max/mean conv sizes
        # - Size
        # - Input / output shapes
        # Reference the ModelMFE (metafeature extractor): https://github.com/modlee-ai/recommender/blob/a86eb715c0f8771bbcb20a624eb20bc6f07d6c2b/data_prep/model_mfe.py#L117
        # In that prior implementation, used the ONNX text representation, and regexes

        return {
            "size": get_model_size(self.torch_model, as_MB=False),
            "output_shape": self.get_output_shape(),
            **self.get_parameter_statistics(self.dataframe),
            **self.get_layer_counts(self.dataframe),
        }

    @abstractmethod
    def get_output_shape(self):
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
        # breakpoint()
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
        # def get_layer_counts(df=dataframe):
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
        # if isinstance(df, pd.Series) or df.shape[1]==1:
        #     return {
        #         statistic:getattr(np, statistic)(df) for statistic in statistics
        #     }
        # else:
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
        pass


class ImageModelMetafeatures(ModelMetafeatures):
    def get_output_shape(
        self,
    ):
        # breakpoint()
        device = next(self.torch_model.parameters()).device
        input_dummy = torch.randn([1, 3, 300, 300]).to(device=device)
        output = self.torch_model(input_dummy)
        return np.array(output.shape[1:])


class ImageClassificationModelMetafeatures(ImageModelMetafeatures):
    pass


class ImageSegmentationModelMetafeatures(ImageModelMetafeatures):
    def get_output_shape(self):
        output = self.torch_model(torch.randn([10, 3, 300, 300]))
        if isinstance(output, dict):
            output = output["out"]
        return np.array(output.shape[1:])


class TextModelMetafeatures(ModelMetafeatures):
    def __init__(self, torch_model, *args, **kwargs):
        input_dummy = torch_model.transform()(modlee.converter.TEXT_INPUT_DUMMY)
        torch_model = torch_model.get_model()

        super().__init__(
            torch_model=torch_model, input_dummy=input_dummy, *args, **kwargs
        )


class ModelEncoder(karateclub.graph2vec.Graph2Vec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.attributed = False

    def save(self, path):
        with open(path, "wb") as _file:
            _file.write(pickle.dumps(self))

    @classmethod
    def from_pkl(cls, path):
        with open(path, "rb") as _file:
            # g2v = pickle.loads(_file.read(path))
            ret = pickle.load(_file)
            ret.attributed = False
            return ret
        # return cls

    
from_modality_task = partial(class_from_modality_task, _class="Model_Metafeatures")

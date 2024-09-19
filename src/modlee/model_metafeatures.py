from abc import abstractmethod
from functools import partial

import numpy as np
import pandas as pd
import karateclub
import pickle

import torch
import modlee
from modlee.config import G2V_PKL
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

        # model_copy = torch_model
        # sample_input_copy = sample_input.clone()  # Use clone to create a copy of the tensor

        # # Move both the model and tensor to the CPU
        # self.model_copy = torch_model# model_copy.to(torch_model.device)
        # self.sample_input_copy = sample_input_copy#.to(torch_model.device)


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
        #self.embedding = self.get_embedding()
        #pass

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

    def get_embedding(self, *args, **kwargs):
        g2v = ModelEncoder.from_pkl(G2V_PKL)
        embd = g2v.infer([self.onnx_nx])[0]
        # breakpoint()
        # embd = g2v.get_embedding([self.onnx_nx])[0]
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
        # device = next(self.torch_model.parameters()).device
        # input_dummy = torch.randn([1, 3, 300, 300]).to(device=device)
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


class TextModelMetafeatures(ModelMetafeatures):
    def __init__(self, torch_model, *args, **kwargs):
        input_dummy = torch_model.transform()(modlee.converter.TEXT_INPUT_DUMMY)
        torch_model = torch_model.get_model()

        super().__init__(
            torch_model=torch_model, input_dummy=input_dummy, *args, **kwargs
        )

class TextClassificationModelMetafeatures(TextModelMetafeatures):
    pass


class TabularModelMetafeatures(ModelMetafeatures):
    pass

class TabularClassificationModelMetafeatures(TabularModelMetafeatures):
    pass

class TimeseriesModelMetafeatures(ModelMetafeatures):
    pass
    
class TimeseriesClassificationModelMetafeatures(TimeseriesModelMetafeatures):
    pass


class ModelEncoder(karateclub.graph2vec.Graph2Vec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.attributed = False

    def save(self, path):
        with open(path, "wb") as _file:
            _file.write(pickle.dumps(self))

    # def infer(self, *args, **kwargs):
        # return super().get_embedding(*args, **kwargs)

    @classmethod
    def from_pkl(cls, path):
        with open(path, "rb") as _file:
            # g2v = pickle.loads(_file.read(path))
            ret = pickle.load(_file)
            ret.attributed = False
            return ret
        # return cls

    
from_modality_task = partial(class_from_modality_task, _class="Model_Metafeatures")

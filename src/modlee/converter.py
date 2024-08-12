"""
The converter module holds the Converter class for converting between different formats of a neural network.
The supported formats include Torch and ONNX.

The Torch formats include:

- Model, the object which contains the forward pass and can be trained
- Code, the model as text code that can be saved to a file and used to rebuild the model

The ONNX formats include:

- Graph, the network represented as a graph with layers as nodes
- Text, the textual description of the graph that is portable and can be rebuilt into a graph
"""
import copy
from importlib.machinery import SourceFileLoader
import os, inspect, sys, logging
import numpy as np
import networkx as nx
import torchsummary
import torch
import onnx2torch
import onnx_graphsurgeon as gs
import onnx
from onnx.tools import net_drawer

ONNX_MINOR_VERSION = int(onnx.__version__.split(".")[1])
import re
import functools
from functools import partial

MODEL_CODE_HEADER = """
import torch, onnx2torch
from torch import tensor
"""

IMAGE_INPUT_DUMMY = torch.randn([10,3,300,300])

TEXT_INPUT_DUMMY = [
    "hello world",
    "the quick brown fox jumps over the lazy dog" * 10,
    "the mitochondria is the powerhouse of the cell",
]


class Converter(object):
    """
    Base object that holds conversion functions.
    """

    
    def torch_model2onnx_graph(
        self, torch_model, input_dummy=None, tmp_onnx_path="./.tmp_model.onnx", modality="tabular", **kwargs
    ):
        """
        Convert a Torch Model to ONNX Graph.
        Note that to reduce the size of the output graph, we set `export_params=False`.
        This and other parameters can be passed as `**kwargs` to `torch.onnx.export`.

        :param torch_model: The Torch Model to convert.
        :param input_dummy: A tensor input to the Torch Model, required for the ONNX parser to determine tensor sizes.
        :param tmp_onnx_path: A placeholder location to save the ONNX graph
        """
        # Keeping gradients on may cause issues, so turn them off

        
        torch_model.eval()
        # TODO - refactor the below
        # Tabular
        if modality=="tabular":
            if input_dummy['x'] is None:
                input_dummy['x'] = torch.randn([10, 3, 300, 300])
            ###
            if isinstance(input_dummy['x'], dict):
                input_dummy['x'] = {key: value.to(device=torch_model.device) for key, value in input_dummy['x'].items()}
                for key, value in input_dummy['x'].items():
                    value.requires_grad = False
            else:
                input_dummy['x'].requires_grad = False
                input_dummy['x'] = input_dummy['x'].to(device=torch_model.device)

            if not isinstance(input_dummy['x'], dict) and hasattr(torch_model, 'device'):
                input_dummy['x'] = input_dummy['x'].to(device=torch_model.device)
        
        elif modality=="timeseries":
            # Timeseries
            '''if input_dummy is None:
                input_dummy = torch.randn([10, 3, 300, 300])
            input_dummy.requires_grad = False
            if hasattr(torch_model, 'device'):
                input_dummy = input_dummy.to(device=torch_model.device)'''
            #input_dummy = next(iter(input_dummy))
            if isinstance(input_dummy, dict):
                for key in input_dummy:
                    if isinstance(input_dummy[key], torch.Tensor):
                        input_dummy[key].requires_grad = False
                        if hasattr(torch_model, 'device'):
                            input_dummy[key] = input_dummy[key].to(device=torch_model.device)
            else:
                input_dummy.requires_grad = False
                if hasattr(torch_model, 'device'):
                    input_dummy = input_dummy.to(device=torch_model.device)

        with torch.no_grad():
            for param in torch_model.parameters():
                param.requires_grad = False
            torch.onnx.export(
                torch_model,
                input_dummy,
                tmp_onnx_path,
                export_params=False,
                opset_version=17,
                input_names=["input_1"],
                output_names=["gemm_1"],
                dynamic_axes={
                    "input_1": [0],
                    "gemm_1": [0],
                },
                **kwargs,
            )

            for param in torch_model.parameters():
                param.requires_grad = True
        torch_model.train()
        # The model we load will have no parameters initialized
        onnx_model = onnx.load(tmp_onnx_path)
        if ONNX_MINOR_VERSION <= 15:
            # Initialize the parameterless model
            onnx_model = self.onnx_parameterless2onnx(onnx_model)
        return onnx_model

    torch2onnx = torch_model2onnx_graph

    
    def torch_model2torch_code(self, torch_model, *args, **kwargs):
        """
        Convert a Torch Model to Torch Code.

        :param torch_model: The Torch Model to convert.
        :return torch_code: The Torch Code.
        """
        onnx_model = self.torch2onnx(torch_model, *args, **kwargs)
        onnx_text = self.onnx2onnx_text(onnx_model)
        model_code = self.onnx_text2code(onnx_text)
        return model_code

    torch2code = torch_model2torch_code

    
    def torch_model2onnx_text(self, torch_model, *args, **kwargs):
        """
        Convert a Torch Model to ONNX Text

        :param torch_model: The Torch Model
        :return onnx_text: The ONNX Text
        """
        return self.onnx2onnx_text(self.torch2onnx(torch_model, *args, **kwargs))

    torch2onnx_text = torch_model2onnx_text

    
    def torch_code2torch_model(
        self, torch_code: str, tmp_model_path="./.tmp_model.py", *args, **kwargs
    ):
        """
        Convert Torch Code into a Torch Model

        :param torch_code: The Torch Code, either as a file path or the raw code text
        :param tmp_model_path: The path to a cache of the code, as a *.py file
        :return: The Torch model.
        """
        # If the input is a path, load the text from the file
        if os.path.exists(torch_code):
            with open(torch_code, "r") as _file:
                torch_code = _file.read()
        # Hold the text in a temporary file
        self.save_code(torch_code, tmp_model_path)
        return self.torch_file2torch_model(tmp_model_path)

    code2torch = torch_code2torch_model

    
    def torch_file2torch_model(self, torch_file):
        """
        Convert a Torch File into a Torch Model

        :param torch_file: The Torch Code as a path
        :return torch_model: The Torch Model
        """
        torch_module = SourceFileLoader(
            fullname="model_module", path=torch_file
        ).load_module()
        return torch_module.Model()

    code_path2torch = torch_file2torch_model

    
    def torch_model2torch_model(self, torch_model, *args, **kwargs):
        """
        Convert a PyTorch model into an equivalent PyTorch model,
        but represented as a graph of layers and operations
        PyTorch -> ONNX -> Code -> ONNX -> PyTorch

        :param torch_model: The Torch model, created normally through code.
        :return: The Torch model, after it has been graphized through ONNX.
        """
        return self.code2torch(self.torch2code(torch_model, *args, **kwargs))

    torch2torch = torch_model2torch_model
    torch2torch_graph = torch_model2torch_model

    
    def onnx_file2torch_model(self, onnx_file, *args, **kwargs):
        """
        Convert an ONNX File to a Torch Model.

        :param onnx_file: The ONNX File as a path.
        :return torch_model: The Torch Model.
        """
        return onnx2torch.convert(onnx_file, *args, **kwargs)

    onnx_path2torch = onnx_file2torch_model

    
    def onnx_file2onnx_graph(self, onnx_file):
        """
        Convert an ONNX File to an ONNX Graph.

        :param onnx_file: The ONNX File as a path.
        :return onnx_graph: The ONNX Graph as a text.
        """
        with open(onnx_file, "r") as _file:
            return self.onnx_text2onnx_graph(_file.read())

    onnx_text_file2onnx = onnx_file2onnx_graph

    
    def onnx_file2torch_model(self, onnx_file):
        """
        Convert an ONNX File to a Torch Model

        :param onnx_file: The ONNX File as a path
        :return torch_model: The Torch Model
        """
        with open(onnx_file, "r") as _file:
            onnx_text = _file.read()
        return self.onnx_text2torch(onnx_text)

    onnx_file2torch = onnx_file2torch_model

    
    def onnx_uninit2torch(self, onnx_graph):
        """
        Convert an uninitialized ONNX Graph to a Torch Model

        :param onnx_graph: The uninitialized ONNX Graph
        :return torch_model: The Torch Model
        """
        return self.onnx_graph2torch_model(self.onnx_parameterless2onnx(onnx_graph))

    
    def onnx_text2torch_code(self, onnx_text):
        """
        Convert ONNX Text to Torch Code

        :param onnx_text: The ONNX Text
        :return torch_code: The Torch Code
        """
        torch_model = self.onnx_text2torch_model(onnx_text)
        torch_code = self.torch_graph2code(torch_model)
        # Deleting model should free some space
        del torch_model
        return torch_code

    onnx_text2code = onnx_text2torch_code

    
    def onnx_text2onnx_graph(self, onnx_text):
        """
        Convert ONNX Text to an ONNX Graph.

        :param onnx_text: The ONNX Text
        :return onnx_graph: The ONNX Graph
        """
        # If on Python 3.12, likely using a newer ONNX
        #breakpoint()
        if ONNX_MINOR_VERSION > 15:
            onnx_text = self.convert_onnx116(onnx_text)
        #breakpoint()
        return onnx.parser.parse_model(onnx_text)

    onnx_text2onnx = onnx_text2onnx_graph

    
    def onnx_text2torch_model(self, onnx_text: bytes):
        """
        Convert ONNX Text to Torch Model.

        :param onnx_text: The ONNX Text as bytes.
        :return: The Torch Model.
        """
        #breakpoint()
        onnx_graph = self.onnx_text2onnx_graph(onnx_text)
        # Load the graph into ONNX Graph Surgeon
        onnx_graph = gs.import_onnx(onnx_graph)
        # Initialize the tensors of the graph
        onnx_graph = self.init_graph_tensors(onnx_graph)
        # Re-export the graph
        onnx_graph = gs.export_onnx(onnx_graph)
        # Convert to Torch
        torch_model = self.onnx_graph2torch_model(onnx_graph)
        return torch_model

    onnx_text2torch = onnx_text2torch_model

    
    def onnx_graph2torch_model(self, onnx_graph, *args, **kwargs):
        """
        Convert an ONNX Graph to a Torch Model.

        :param onnx_graph: The ONNX Graph object.
        :return torch_model: The Torch Model.
        """
        # Handle conversion for newer ONNX versions
        # TODO - try to remove the try/except block
       # breakpoint()
        try:
            return onnx2torch.convert(onnx_graph, *args, **kwargs)
        except:
            pass
        if ONNX_MINOR_VERSION >= 16:
            try:
                onnx_text = self.onnx_graph2onnx_text(onnx_graph)
                _onnx_graph = self.onnx_text2onnx_graph(onnx_text)
                return onnx2torch.convert(_onnx_graph, *args, **kwargs)
            except:
                onnx_text = self.onnx_graph2onnx_text(onnx_graph)
                _onnx_graph = self.onnx_text2onnx_graph(onnx_text)
                _onnx_graph = self.onnx_parameterless2onnx(_onnx_graph)
                return onnx2torch.convert(_onnx_graph, *args, **kwargs)   
        else:   
            return onnx2torch.convert(onnx_graph, *args, **kwargs)   #converter.onnx_graph2torch_model(onnx_graph) fails here

    onnx2torch = onnx_graph2torch_model

    
    def onnx_graph2onnx_text(self, onnx_graph, remove_identity=False):
        """
        Convert an ONNX Graph to ONNX Text

        :param onnx_graph: The ONNX Graph to convert
        :param remove_identity: Whether to remove Identity layers in the output text
        :return: The ONNX Text representation
        """

        def get_inner_string(s, _start, _end):
            """
            TODO rewrite the Converter().get_inner_string() to be this simple,
            and handle the string splitting in a wrapper or the methods that use it
            """
            s = s[s.find(_start) + len(_start) :]
            s = s[: s.rfind(_end)]
            return s

        onnx_str = onnx.printer.to_text(onnx_graph)
        onnx_str = onnx_str.split("\n")
        output_var = "None"
        n_lines = len(onnx_str)
        layer_name_type_dict = {}

        # Regex expression to track floating point number calls"
        permitted_onxx_float_pattern = r"^(\{)?-?\d*(_\d*)*(e-?\d*)?(\})?[-\d>](,)?$"

        for line_ctr, onnx_uninit_line in enumerate(onnx_str):
            # Skip header
            if line_ctr < 6:
                continue

            # Replace characters that cannot be parsed by onnx.parser.parse_model
            unparseable_chars = [".", ":", "/"]

            # Creating a lsit of string elements to parse
            for unparseable_char in unparseable_chars:
                # Tracking decmimal point separately to enable parsing of floating point numbers
                # Simply converting other characters to '_' to facilitate parsing.
                onnx_uninit_line = onnx_uninit_line.replace(unparseable_char, "_")

            # For NASLib models, handle unparseable characters in e.g. makrograph-edge(7,8)_...
            # Handles the dash, comma, and parentheses
            onnx_uninit_line = re.sub(
                "makrograph-edge\((\d*),(\d*)\)_",
                "makrograph_edge_\\1_\\2_",
                onnx_uninit_line,
            )

            # Refactor malformed boolean layers
            if "bool" in onnx_uninit_line:
                onnx_uninit_line = self.refactor_bool_layer(onnx_uninit_line)

            # Refactor inf to large value:
            if "inf" in onnx_uninit_line:
                onnx_uninit_line = self.refactor_inf(onnx_uninit_line)
            # Case: the line is defining a Constant float value that should keep the '.' within brackets {}
            # e.x. const_output_0 = Constant <value = float {0_08}>
            # '0_08' should be reverted back to '0.08'

            onnx_uninit_line_as_list = onnx_uninit_line.split(" ")
            for idx, onnx_str_item in enumerate(onnx_uninit_line_as_list):
                # Only replacing decimal point with string if pattern matches, allows capture of floating point values as parameters
                if re.match(permitted_onxx_float_pattern, onnx_str_item):
                    onnx_uninit_line_as_list[idx] = onnx_str_item.replace("_", ".")
                else:
                    continue

            onnx_uninit_line = " ".join(onnx_uninit_line_as_list)

            if " Constant " in onnx_uninit_line:
                onnx_uninit_line = self.convert_float(onnx_uninit_line)

            # Found line with output variable, which must be a non-number
            # e.g. "191" is not valid, so we override it with "output_var"
            if "=>" in onnx_uninit_line:
                output_var = get_inner_string(onnx_uninit_line, "=>", "{").strip()
                output_var = get_inner_string(output_var, "]", ")").strip()
                onnx_uninit_line = onnx_uninit_line.replace(
                    f"] {output_var}) {{", f"] output_var) {{"
                )
            elif line_ctr < (n_lines - 1):
                # Add the layer name to the respective layer type in the "counter" dictionary
                layer_name, _, layer_type = onnx_uninit_line.split()[:3]#####
                # split_line = onnx_uninit_line.split()
              
                # if len(split_line) >= 3:
                #     layer_name, _, layer_type = split_line[:3]
                # else:
                #     layer_name = "unknown_layer"
                #     layer_type = "unknown_type"

                if layer_type not in layer_name_type_dict:
                    layer_name_type_dict.update({layer_type: [layer_name]})
                else:
                    layer_name_type_dict[layer_type].append(layer_name)

            onnx_str[line_ctr] = onnx_uninit_line

        onnx_str = "\n".join(onnx_str)
        # Replace the output variable with the generic 'output_var'
        onnx_str = onnx_str.replace(f"{output_var} =", f"output_var =")

        # Refactor any variables with leading numbers
        onnx_str = self.refactor_leading_number(onnx_str)

        for layer_type, layer_names in layer_name_type_dict.items():
            for layer_idx, layer_name in enumerate(layer_names):
                if layer_name.isdigit():
                    continue
                onnx_str = onnx_str.replace(
                    layer_name, f"{layer_type.lower()}_output_{layer_idx:04d}"
                )

        if remove_identity:
            onnx_str = self.remove_identity(onnx_str)
        return onnx_str

    onnx2onnx_text = onnx_graph2onnx_text

    def filter_node(self, x):
        """
        Returns whether this is a non-layer node to filter out
        Checks for substrings in the node name that indicate that it is not a layer.

        :param x: The NetworkX node to check.
        :return: Whether the node contains a substring indicating that it should be filtered as a non-layer.
        """
        return "onnx::" in x or "Identity" in x or "fc." in x

    def prune_onnx_nx(self, onnx_nx):
        """
        Prune an ONNX NetworkX graph to just the layer nodes.

        :param onnx_nx: The ONNX NetworkX graph to prune.
        :return: The pruned ONNX NetworkX graph.
        """
        nodes_to_prune = [k for k in onnx_nx.nodes.keys() if self.filter_node(k)]
        # help(onnx_nx.remove_node)
        onnx_nx_layers_only = copy.deepcopy(onnx_nx)
        for node in nodes_to_prune:
            onnx_nx_layers_only.remove_node(node)
        return onnx_nx_layers_only
            
    
    def onnx_graph2onnx_nx(self, onnx_graph, prune=True):
        """
        Convert an ONNX graph to ONNX NetworkX.

        :param onnx_graph: The ONNX graph.
        :param prune: Whether to prune the NetworkX to just layer nodes, defaults to True
        :return: The ONNX NetworkX graph.
        """
        if ONNX_MINOR_VERSION <= 15:
            onnx_graph = self.onnx_parameterless2onnx(onnx_graph)
        onnx_pydot = onnx.tools.net_drawer.GetPydotGraph(onnx_graph.graph)
        onnx_pydot.set_name("onnx_graph")
        onnx_nx = nx.nx_pydot.from_pydot(onnx_pydot)
        if prune:
            onnx_nx = self.prune_onnx_nx(onnx_nx)
        return onnx_nx

    def index_nx(self, onnx_nx):
        """
        Index an ONNX NetworkX graph, by replacing the node labels with their indices.

        :param onnx_nx: The ONNX NetworkX to index.
        :return: The ONNX NetworkX ,indexed. The function modifies the graph in-place and the return value should be unnecessary.
        """
        relabel_dict = {}
        for n, node in enumerate(onnx_nx.nodes(data=True)):
            relabel_dict.update({node[0]: n})
        for k, v in relabel_dict.items():
            nx.relabel_nodes(
                onnx_nx,
                {k: v},
                copy=False,
            )
        return onnx_nx

    def remove_identity(self, onnx_text):
        """
        Remove identity layers in ONNX Text.

        :param onnx_text: The ONNX Text.
        :return: The ONNX Text stripped of identity layers.
        """

        # Patterns to find 'identity_output_xxxx' assignments and their usage
        # pattern_assignment = re.compile(r'identity_output_(\d{4})\s*=\s*Identity\s*\((onnx__Conv_\d+)\)')
        pattern_assignment = re.compile(
            r"identity_output_(\d{4})\s*=\s*Identity\s*\(([^)]+)\)"
        )

        assignments = {
            match[0]: match[1] for match in pattern_assignment.findall(onnx_text)
        }

        # Remove the assignment lines for 'identity_output_xxxx'
        onnx_text = re.sub(pattern_assignment, "", onnx_text)

        # Replace each instance of 'identity_output_xxxx' with its assigned value
        for identity_number, actual_value in assignments.items():
            onnx_text = onnx_text.replace(
                f"identity_output_{identity_number}", actual_value
            )

        # Remove multiple spaces and replace them with a single space
        onnx_text = re.sub(r" +", " ", onnx_text)
        # Remove chunks of blank space (multiple newlines)
        onnx_text = re.sub(r"\n\s*\n", "\n", onnx_text)
        return onnx_text

    def refactor_bool_layer(self, input_str):
        """
        Refactor boolean layers to the correct number of input elements
        The onnx.printer.to_text() function seems to remove any inputs that the parser would use.
        For example, an int layer is defined like:
        constant_output_0006 = Constant <value = int64[4] {3,12,-1,-1}> ()

        From:
        constant_output_0005 = Constant <value = bool[1,1,3,3]___> ()
        To:
        constant_output_0005 = Constant <value = bool[1,1,3,3] {0,0,0,0,0,0,0,0,0}> ()

        :param input_str: The string with boolean layers.
        :return: The string with boolean layers properly refactored.
        """
        if "bool" not in input_str:
            return input_str
        bool_dim = re.search("bool\[(?P<bool_dim>.*)\]", input_str)
        if bool_dim:
            bool_dim = bool_dim["bool_dim"]
            n_elements = np.prod([int(_b) for _b in bool_dim.split(",")])
            bool_ending = r"]"
        else:
            # Handle case where there is no bool dim - should just be one value
            bool_dim = ""
            n_elements = 1
            bool_ending = "bool"
        input_arg_list = f'{{{",".join("0"*n_elements)}}}'
        input_str = re.sub(
            f"{bool_ending}(.*)>", f"{bool_ending} {input_arg_list}>", input_str
        )
        input_str = re.sub(r"/s{2,}", " ", input_str)
        return input_str

    def refactor_inf(self, input_str, large_value="99999999"):
        """Replace 'inf' with a large value because the parser cannot handle infs

        :param input_str: The string with 'inf'.
        :param large_value: A suitably large value to replace 'inf' with, defaults to "99999999".
        :return: The string with 'inf' refactored with a large value.

        """
        if not isinstance(large_value, str):
            large_value = str(large_value)
        if "inf" not in input_str:
            return input_str
        return re.sub("inf", str(large_value), input_str)

    def refactor_leading_number(self, input_str):
        """
        Refactor variables with leading numbers which are not parseable,
        e.g. 0_model_fc_weight -> model_0_fc_weight

        :param input_str: The string with variables with leading numbers.
        :return: The string with number-lead variables refactored.
        """
        return re.sub("([\s(,]+)(\d+)_*([a-zA-Z0-9]*)[\s_=]", "\\1\\3_\\2_", input_str)

    # Helper functions when converting from ONNX
    def init_graph_tensors(
        self, onnx_gs_graph, tensor_init_fn=partial(np.random.normal, scale=0.01)
    ):
        """
        Initialize the graph's tensors, in place (you do not need to use the return value)
        The input should be an ONNX graph exported from torch without parameters, i.e.
        torch.onnx.export(..., export_params=False).

        Identity layers get special treatment; they must be initialized for onnx2torch,
        but their target shape is nested within its inputs.

        Example usage:
        import onnx_graphsurgeon as gs
        graph = gs.import_onnx(path/to/uninitialized/model.onnx)
        Converter().init_graph_tensors(graph)

        :param onnx_gs_graph: The ONNX GraphSurgeon Graph with uninitialized weights.
        :param tensor_init_fn: The initialization function to use for the graph's weights, defaults to `np.random.normal(scale=0.01)`
        :return: The ONNX GraphSurgeon Graph with tensors initialized. The function modifies the graph in place, so assigning the return to the graph is not necessary.
        """
        graph_tensors = onnx_gs_graph.tensors()
        for tensor_key, tensor_value in graph_tensors.items():
            # Skip initializing any tensors that are already Constants
            if "constant" in str(type(tensor_value)).lower():
                if "identity" not in tensor_key.lower():
                    continue
            # Skip tensors that are inputs/outputs and should be kept as Variables
            # Converting these to constants would essentially "freeze" the network
            # into deterministic outputs
            if any([_substr in tensor_key for _substr in ["input", "output"]]):
                if "identity" not in tensor_key.lower():
                    if "constant" not in tensor_key.lower():
                        continue

            if isinstance(tensor_value, (int, float)):
                value_shape = (1,)
            elif "identity" in tensor_key:
                value_shape = tensor_value.inputs[0].inputs[0].shape
            else:
                value_shape = tensor_value.shape
            if value_shape is None:
                continue
            if len(value_shape) == 0:
                continue
            if isinstance(value_shape[0], str):
                if "dynamic_axes" in value_shape[0]:
                    continue
            # print(value_shape, type(value_shape))

            tensor_value.to_constant(
                values=tensor_init_fn(
                    size=value_shape,
                ).astype(np.float32)
            )
        return onnx_gs_graph

    def init_onnx_params(self, onnx_graph):
        """
        Initialize a parameterless ONNX Graph

        :param onnx_graph: The ONNX Graph.
        :return onnx_graph: The ONNX Graph with initialized parameters.
        """
        onnx_graph = self.init_onnx_tensors(onnx_graph)
        return gs.export_onnx(onnx_graph)

    onnx_parameterless2onnx = init_onnx_params

    def init_onnx_tensors(self, onnx_graph):
        """
        Initialize the tensors of an ONNX Graph

        :param onnx_graph: The ONNX Graph
        :return onnx_graph: The ONNX Graph with initalized tensors
        """
        onnx_graph = gs.import_onnx(onnx_graph)
        return self.init_graph_tensors(onnx_graph)

    onnx2onnx_gs = init_onnx_tensors

    def save_torch(self, torch_model, filepath):
        """
        Save a PyTorch model's code representation as a `.py` file.

        :param torch_model: The Torch Model to save.
        :param filepath: The path to where the model should be saved, should end in `.py`.
        """
        self.save_code(self.torch2code(torch_model), filepath=filepath)

    def save_code(self, torch_code, filepath):
        """
        Save a PyTorch model's sring representation as a `.py` file.

        :param torch_model: The Torch Model to save.
        :param filepath: The path to where the model should be saved, should end in `.py`.
        """
        with open(filepath, "w") as _file:
            _file.write(torch_code)

    # Helper functions for creating code representations from a model
    def get_inner_string(self, input_str, _start, _end, return_only_single_value=True):
        """
        Get the inner string between a start and end sequence.
        If there are multiple potential inner strings (e.g. if there are multiple instances
        of the start and/or end sequences), returns the longest valid substring.

        :param input_str: The string to extract from.
        :param _start: The start of the string sequence.
        :param _end: The end of the string sequence.
        :param return_only_single_value: Whether to return only the first whitespace-split item in the found sequence, defaults to Truedefaults to True.
        :return: The inner string, or None if the _start and _end sequences could not be found.
        """
        # If cannot find either substring limiters, return None
        if any([idx == -1 for idx in [input_str.find(_start), input_str.find(_end)]]):
            return None

        input_str = input_str[input_str.find(_start) + len(_start) :]
        input_str = input_str[: input_str.rfind(_end)]
        # valid attributes should just be one string
        if return_only_single_value:
            if len(input_str.split()) == 1:
                return input_str
            else:
                return None
        else:
            return input_str

    def get_attr_name(self, input_str):
        """
        Get the variable name of an object attribute from a string.

        The input string to this function should be a line from the forward pass of a model
        converted from onnx2torch, e.g. :
        model_conv1_conv = getattr(self, "model/conv1/Conv")(input_1);  input_1 = None
        Will retrieve "model/conv1/Conv".

        :param input_str: The input string from which to get the attribute from.
        :return: The attribute name.
        """
        attr_name = self.get_inner_string(
            input_str, _start='getattr(self, "', _end='")'
        )
        # Catch a case where the attribute is directly accessed,
        # e.g. self.LogSoftmax -> retrieve "LogSoftmax"
        if attr_name is None:
            attr_name = self.get_inner_string(input_str, _start="self.", _end="(")
        return attr_name

    def get_model_attr_on_line(self, model, line):
        """
        Get attributes as {attribute_name : attribute_object} pairs

        :param model: The model to retrieve attributes from.
        :param line: The line that has the attribute to retrieve.
        :return: A dictionary of the {attribute_name : attribute_object}.
        """
        attr_name = self.get_attr_name(line)
        if attr_name:
            try:
                return {attr_name: getattr(model, attr_name)}
            except:
                return {}

    def get_model_attrs_in_forward(self, model):
        """
        Get all of the attributes from a model's forward pass.

        :param model: The model to get attributes from.
        :return: A dictionary of { attribute_name : attribute_object } pairs
        """
        fwd_source = inspect.getsource(model.forward)
        fwd_lines = fwd_source.split("\n")
        model_attrs = {}
        for fwd_line in fwd_lines[1:]:
            model_attr = self.get_model_attr_on_line(model, fwd_line)
            if model_attr:
                model_attrs.update(model_attr)
        return model_attrs

    def get_params_for_attr(self, model_attr):
        """
        Get the parameters required to initialize an attribute object,
        e.g. convolutional filter sizes / strides, frozen directly from the object.

        :param model_attr: The attribute object to get parameters for.
        :return: The parameters to reinitialize the object in the same state.
        """
        attrs_to_skip = ["bias"]
        attr_kwargs = dict(inspect.signature(model_attr.__init__).parameters)
        attr_params = {}

        for attr_key, attr_val in attr_kwargs.items():
            if attr_key in attrs_to_skip:
                continue
            if hasattr(model_attr, attr_key):
                model_attr_value = getattr(model_attr, attr_key)
                # Convert potentially large tensors to constructors to reduce size
                if isinstance(model_attr_value, torch.Tensor):
                    model_attr_value = self.tensor2init_code(
                        model_attr_value,
                        tensor_type="randn" if "initial" in attr_key else None,
                    )
                attr_params.update({attr_key: model_attr_value})
        if hasattr(model_attr, "state_dict"):
            model_state_dict = model_attr.state_dict()
            # Do this only for initializers because the output file could get large
            model_state_dict = {
                k: self.tensor2init_code(v, tensor_type="randn")
                for k, v in model_state_dict.items()
                if "initial" in k
            }
            if len(model_state_dict) > 0:
                attr_params.update(model_state_dict)
        # If the model_attr is a torch module that has a state_dict, add it
        return attr_params

    def get_type_string(self, obj) -> str | None:
        """
        Get the type of an object as a string.
        TODO - refactor this as a regex

        :param obj: The object.
        :return: The type of the object as a string.
        """
        return self.get_inner_string(str(obj.__class__), _start="<class '", _end="'>")

    def get_init(self, model) -> str:
        """
        Get the code for a model's __init__() constructor function.

        :param model: The model.
        :return: The model's __init__() function as a string.
        """
        assert model is not None
        model_attrs = self.get_model_attrs_in_forward(model)
        model_attrs.update(
            {
                onnx_attr: getattr(model, onnx_attr)
                for onnx_attr in dir(model)
                if "onnx" in onnx_attr
            }
        )
        if hasattr(model, "initializers"):
            model_attrs.update({"initializers": getattr(model, "initializers")})
        init_attrs = []

        for model_attr_name, model_attr in model_attrs.items():
            attr_params = self.get_params_for_attr(model_attr)
            init_attrs.append(
                (model_attr_name, self.get_type_string(model_attr), attr_params)
            )
        spacer = "    "
        spacer = "    "
        init_lines = [
            "",
            f"{spacer}def __init__(self):",
            f"{spacer*2}super().__init__()",
        ]
        for init_attr in init_attrs:
            torch.set_printoptions(threshold=np.inf)
            # Convert potentially large tensors to constructors to reduce size
            kwarg_str = self.dict2code(init_attr[2])
            init_line = f"{spacer*2}setattr(self,'{init_attr[0]}', {init_attr[1]}(**{kwarg_str}))"
            init_lines.append(init_line)

            # If the attribute is the ONNX initializer,
            # 1) remove the kwargs to the module constructor call
            # 2) append the code to register the initializer states
            if "initial" in init_attr[0]:
                # remove the just-appended line
                init_line = init_lines.pop().replace("**" + kwarg_str, "")
                init_lines.append(init_line)
                initializer_init_str = self.get_init_module_state_dict_str(
                    f"self.{init_attr[0]}", kwarg_str
                )
                init_lines.append(initializer_init_str)
        init_code = "\n".join(init_lines)
        return init_code

    def get_init_module_state_dict_str(
        self, module_name_str: str, state_dict_str: str, indent_level=2
    ):
        """
        Return a string that, when called with exec(), will initialize the torch module's state dictionary.
        The module will likely be a freshly initialized module with an empty state dict.
        This uses register_buffer to add unexpected keys to the state dict.

        :param module_name: The variable name of the module to be initialized, as a string. Must have already been initialized.
        :param state_dict: A string representation of the state_dict
        :param indent_level: The amount of indents (4 whitespaces) to prepend to each line, defaults to 2 for use in a class function
        :return: Code text to initialize the module's state dict.
        """
        spacer = "    "
        ret_str = ""
        ret_str += f"{spacer*indent_level}init_state_dict = {state_dict_str}\n"
        ret_str += f"{spacer*indent_level}for k,v in init_state_dict.items():\n"
        ret_str += f"{spacer*(indent_level+1)}{module_name_str}.register_buffer(k,v)\n"
        return ret_str

    def dict2code(self, kwarg_dict):
        """
        Converts a dictionary into a code string that, when called with exec(), rebuilds the dictionary.

        :param kwarg_dict: The dictionary to convert.
        :return: A code string to create the dictionary.
        """
        kwarg_list = []
        for kwarg_key, kwarg_value in kwarg_dict.items():
            formatted_kwarg_value = kwarg_value
            if isinstance(formatted_kwarg_value, str):
                try:
                    exec(formatted_kwarg_value)
                except:
                    formatted_kwarg_value = f"'{formatted_kwarg_value}'"

            kwarg_list.append(f"'{kwarg_key}':{formatted_kwarg_value}")
            import os

            with open(os.path.expanduser("~/kwargs.txt"), "a") as _file:
                _file.write(kwarg_list[-1])
                _file.write("\n")
        return f"{{{','.join(kwarg_list)}}}"
        return

    def tensor2init_code(self, input_tensor, tensor_type: str = None):
        """
        Converts a monovalue tensor (len(set(tensor))==1) to a string representation of its initialization.
        Minifies potentially large from their explicit definition to simply 'tensor.ones((x,y))*values'.
        If `tensor_type` is provided, this function will force-convert a non-uniform tensor to an
        initialization string for a tensor of that type (e.g. 'randn','ones','zeros').

        :param input_tensor: The tensor to convert
        :param tensor_type: The tensor type to convert to, from ['randn','zeros','ones']. Will try to auto-detect if not provided.
        :return: A code string to create the tensor.
        """
        if not isinstance(input_tensor, torch.Tensor):
            return input_tensor
        tensor_set = set(input_tensor.flatten().tolist())
        if len(tensor_set) > 1:
            torch.set_printoptions(threshold=torch.inf)
            if tensor_type is None:
                # If no type override is defined, get the full class of the tensor
                # to rebuild the tensor into one of the same class
                full_tensor_class = re.search(
                    "'(?P<tensor_class>.*)'", str(type(input_tensor))
                )["tensor_class"]
                return f"{full_tensor_class}({input_tensor.numpy().tolist()})"
            else:
                # If type override is defined, create a tensor in the shape
                # of the input
                full_tensor_class = f"torch.{tensor_type}"
                return f"{full_tensor_class}({input_tensor.shape})"
        else:
            tensor_value = tensor_set.pop()
            tensor_shape = tuple(input_tensor.shape)
            if abs(tensor_value) < 0.00000001:
                output_str = f"torch.zeros({tensor_shape})"
            else:
                output_str = f"torch.ones({tensor_shape})*{tensor_value}"
            return output_str.replace(" ", "")

    def get_forward(self, model) -> str:
        """
        Get a model's forward() pass as code.

        :param model: The model.
        :return: The forward() code.
        """
        spacer = "    "
        fwd_lines = inspect.getsource(model.forward).split("\n")
        for i, fwd_line in enumerate(fwd_lines[1:]):
            if "self.Tile" in fwd_line:
                fwd_line = self.cast_tile_layer(fwd_line)
            elif "self.Gather" in fwd_line:
                fwd_line = self.cast_gather_layer(fwd_line)
            fwd_lines[i + 1] = f"{spacer}{fwd_line}"

        return "\n".join(fwd_lines)

    def get_model_code(self, model) -> str:
        """
        Retrieve the model's string representation, which includes its constructor (__init__) and forward pass.
        The code, when imported as a module or called with exec(), will rebuild the model object.

        :param model: The model.
        :return: The code for the entire model module.
        """
        model_init_code = self.get_init(model)
        model_fwd_code = self.get_forward(model)
        spacer = "    "
        model_code = f"""
import torch, onnx2torch
from torch import tensor
class Model(torch.nn.Module):
{spacer}{model_init_code}
{spacer}{model_fwd_code}
"""
        return model_code

    torch_graph2code = get_model_code

    def cast_tile_layer(self, input_str):
        """
        Cast variables in a Tile ONNX layer to the required types.

        :param input_str: The string of the Tile layer.
        :return: The layer string with types properly cast.
        """
        return re.sub(
            "(Tile.*), (.*)\)",
            "\\1, list(\\2.type(torch.int64).cpu().numpy()))",
            input_str,
        )

    def cast_gather_layer(self, input_str):
        """
        Cast variables in a Gather ONNX layer to the required types.

        :param input_str: The string of the Gather layer.
        :return: The string with types propery cast.
        """
        return re.sub("(Gather.*), (.*)\)", "\\1, \\2.type(torch.int64))", input_str)

    def convert_onnx116(self, onnx_text):
        """
        Convert ONNX graph text generated with ONNX 1.16, which requires modifications
        to be parseable by onnx.parser.

        :param onnx_text: The
        ONNX graph text to convert.
        :return: The ONNX graph text converted to a format parseable by onnx.parser.
        """
        if not isinstance(onnx_text, str):
            onnx_text = str(onnx_text)
        return (
            onnx_text.replace("_ ", " ")
            .replace(" ints =", " =")
            .replace(" int =", " =")
            .replace(" float =", " =")
            .replace(" tensor ", " ")
            .replace(" string =", " =")
        )

    def convert_float(self, onnx_text):
        return re.sub("(.*)float(.*)_(.*)", "\\1float\\2.\\3", onnx_text)

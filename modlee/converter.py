from importlib.machinery import SourceFileLoader
import inspect
import numpy as np

import torch
import torchsummary
import onnx2torch
import onnx_graphsurgeon as gs
import onnx

# TODO - cache models for "longer" pipelines
# e.g. torch2code do not need to convert to onnx every time

MODEL_CODE_HEADER = """
import torch, onnx2torch
from torch import tensor
"""

class Converter(object):
    def __init__(self):
        pass

    def torch2onnx(self, torch_model, input_dummy=torch.randn([10, 3, 300, 300]), tmp_onnx_path="./tmp_model.onnx"):
        """
        Convert a PyTorch model to onnx.

        Args:
            torch_model (_type_): A PyTorch model to convert
            tmp_onnx_path: The filename to cache the ONNX model to

        Returns:
            ModelProto: The ONNX model
        """
        # TODO - generalize this input_dummy
        # This is a placeholder for ResNet-based models,
        # and probably other models that take 3-channel images as inputs
        # input_dummy = torch.randn([10, 3, 300, 300])
        torch.onnx.export(
            torch_model,
            input_dummy,
            tmp_onnx_path)
        return onnx.load(tmp_onnx_path)

    def onnx_path2torch(self, onnx_path, *args, **kwargs):
        """
        Retrieve an ONNX model as a PyTorch model

        Args:
            onnx_path (_type_): The path to the ONNX model to retrieve (e.g. model.onnx)

        Returns:
            _type_: The PyTorch model
        """
        return onnx2torch.convert(onnx_path, *args, **kwargs)

    def onnx2torch(self, onnx_model, *args, **kwargs):
        """
        Convert an ONNX model to a PyTorch model

        Args:
            onnx_model (_type_): The ONNX model object to convert

        Returns:
            _type_: The PyTorch model
        """
        return onnx2torch.convert(onnx_model, *args, **kwargs)

    def torch2code(self, torch_model, *args, **kwargs):
        """
        Convert a PyTorch model to a code representation

        Args:
            torch_model (_type_): The PyTorch model to convert

        Returns:
            str: The string representation of the code
        """
        return self.get_model_code(
            self.onnx2torch(
                self.torch2onnx(
                    torch_model, *args, **kwargs)))

    def code2torch(self, torch_code, tmp_model_path="./tmp_model.py", *args, **kwargs):
        """
        Convert a code representation to a PyTorch model

        Args:
            torch_code (str): The string representation of the code
            tmp_model_path (str): Where to cache the code as a .py file

        Returns:
            nn.Module: The PyTorch model
        """
        self.save_code(torch_code, tmp_model_path)
        model_module = SourceFileLoader(
            'model_module',
            tmp_model_path).load_module()
        return model_module.Model()

    def torch2torch(self, torch_model, *args, **kwargs):
        """
        Convert a PyTorch model into an equivalent PyTorch model,
        but represented as a graph of layers and operations
        PyTorch -> ONNX -> Code -> ONNX -> PyTorch

        Args:
            torch_model (_type_): The PyTorch model to convert

        Returns:
            _type_: _description_
        """
        return self.code2torch(
            self.torch2code(
                torch_model, *args, **kwargs))
    # This function name may be more informative
    torch2torch_graph = torch2torch

    def save_torch(self, torch_model, filepath):
        """
        Save a PyTorch model's code representation as a .py file

        Args:
            torch_model (_type_): The PyTorch model to save
            filepath (_type_): Where to save
        """
        self.save_code(self.torch2code(torch_model), filepath=filepath)

    def save_code(self, torch_code, filepath):
        """
        Save a PyTorch model string representation to a .py file

        Args:
            torch_code (_type_): The string representation of the PyTorch model
            filepath (_type_): Where to save
        """
        with open(filepath, 'w') as _file:
            _file.write(torch_code)

    """
    Below are helper functions for creating code representations from a model
    """

    def get_inner_string(self, s, _start, _end):
        """
        Retrieve an inner string between a start and end sequence
        If there are multiple potential inner strings (e.g. if there are multiple instances
        of the start and/or end sequences), returns the longest valid substring
        Returns None if no inner string is found

        Args:
            s (_type_): The full string to retrieve from
            _start (_type_): The start substring
            _end (_type_): The end substring

        Returns:
            _type_: The inner string 
        """
        s = s[s.find(_start)+len(_start):]
        s = s[:s.rfind(_end)]
        # valid attributes should just be one string
        if len(s.split()) == 1:
            return s
        else:
            return None

    def get_attr_name(self, s):
        """
        Parse the name of an object attribute referred to on a string
        
        The input string to this function should be a line from the forward pass of a model
        converted from onnx2torch, e.g. :
        model_conv1_conv = getattr(self, "model/conv1/Conv")(input_1);  input_1 = None
        
        This function will retrieve "model/conv1/Conv"

        Args:
            s (_type_): The string to retrieve the attribute from

        Returns:
            _type_: The attribute name
        """
        attr_name = self.get_inner_string(s,
                                          _start='getattr(self, "',
                                          _end='")',
                                          )
        
        # Catch a case where the attribute is directly accessed,
        # e.g. self.LogSoftmax -> retrieve "LogSoftmax"
        if attr_name is None:
            attr_name = self.get_inner_string(s,
                                              _start='self.',
                                              _end='(',
                                              )
        return attr_name

    def get_model_attr_on_line(self, model, line):
        """
        Return attributes as {attribute_name : attribute_object} pairs

        Args:
            model (_type_): The model to retrieve attributes from
            line (str): The line that has the attribute to retrieve

        Returns:
            dict: {attribute_name : attribute_object}
        """
        attr_name = self.get_attr_name(line)
        if attr_name:
            try:
                return {attr_name: getattr(model, attr_name)}
            except:
                return {}

    def get_model_attrs_in_forward(self, model):
        """
        Retrieve all of the attributes from a model's forward pass
        This should yield all of the attributes required to do a forward pass

        Args:
            model (_type_): The model

        Returns:
            _type_: A dictionary of { attribute_name : attribute_object } pairs
        """
        fwd_source = inspect.getsource(model.forward)
        fwd_lines = fwd_source.split('\n')
        model_attrs = {}
        for fwd_line in fwd_lines[1:]:
            model_attr = self.get_model_attr_on_line(model, fwd_line)
            if model_attr:
                # model_attrs.append(model_attr)
                model_attrs.update(model_attr)
        return model_attrs

    def get_params_for_attr(self, model_attr):
        """
        Retrieve the parameters required to initialize an attribute object
        e.g. convolutional filter sizes / strides

        Args:
            model_attr (_type_): The attribute object

        Returns:
            _type_: A dictionary of { parameter_key : parameter_default } values
        """
        attrs_to_skip = ['bias']
        attr_kwargs = dict(inspect.signature(model_attr.__init__).parameters)
        attr_params = {}
        for attr_key, attr_val in attr_kwargs.items():
            if attr_key in attrs_to_skip:
                continue
            # if attr_val.default==inspect._empty:
            # if True:
            if hasattr(model_attr, attr_key):
                attr_params.update({attr_key: getattr(model_attr, attr_key)})
        return attr_params

    def get_type_string(self, obj) -> str | None:
        """
        Retrieve the type of an object as a string

        Args:
            obj (_type_): The object

        Returns:
            str | None: The type as a string
        """
        return self.get_inner_string(
            str(obj.__class__),
            _start="<class '",
            _end="'>"
        )

    def get_init(self, model) -> str:
        """
        Retrieve the code for a model's __init__() function

        Args:
            model (_type_): The model

        Returns:
            str: An executable __init__ string
        """
        model_attrs = self.get_model_attrs_in_forward(model)

        init_attrs = []

        for model_attr_name, model_attr in model_attrs.items():
            attr_params = self.get_params_for_attr(model_attr)
            init_attrs.append((model_attr_name,
                               self.get_type_string(model_attr), attr_params))
        # init_code = 'def __init__(self):\n'
        spacer = "    "
        spacer = "    "
        init_lines = [
            '',
            f'{spacer}def __init__(self):',
            f'{spacer*2}super().__init__()',
        ]
        for init_attr in init_attrs:
            init_line = f"{spacer*2}setattr(self,'{init_attr[0]}', {init_attr[1]}(**{init_attr[2]}))"
            init_lines.append(init_line)
        init_code = '\n'.join(init_lines)
        return init_code

    def get_forward(self, model) -> str:
        """
        Retrieve a model's forward pass as code

        Args:
            model (_type_): The model

        Returns:
            str: The forward pass as a string
        """
        spacer = "    "
        fwd_lines = inspect.getsource(model.forward).split('\n')
        for i, fwd_line in enumerate(fwd_lines[1:]):
            fwd_lines[i+1] = f"{spacer}{fwd_line}"
        return '\n'.join(fwd_lines)

    def get_model_code(self, model) -> str:
        """
        Retrieve the model's string representation, which includes its constructor (__init__) and forward pass
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
    
    """
    Below are helper functions for importing from torch
    """
    
    def init_graph_tensors(self, graph, tensor_init_fn=np.random.uniform):
        """
        Initialize the graph's tensors, in place (you do not need to use the return value)
        The input should be an ONNX graph exported from torch without parameters, i.e.
        torch.onnx.export(..., export_params=False)
        This enables exporting to torch
        
        Example usage:
        import onnx_graphsurgeon as gs
        graph = gs.import_onnx(path/to/uninitialized/model.onnx)
        Converter().init_graph_tensors(graph)

        Args:
            graph (_type_): _description_
        """
        graph_tensors = graph.tensors()
        for tensor_key,tensor_value in graph_tensors.items():
            if isinstance(tensor_value, (int,float,)):
                value_shape = (1,)
            else:
                value_shape = tensor_value.shape
            if value_shape is None: continue
            tensor_value.to_constant(
                values=tensor_init_fn(size=value_shape))
        return graph

    def onnx_parameterless2onnx(self, onnx_path):
        graph = gs.import_onnx(
            onnx.load(onnx_path))
        graph = self.init_graph_tensors(graph)
        return gs.export_onnx(graph)
        
    def onnx_uninit2torch(self, onnx_path):
        return self.onnx2torch(
            self.onnx_parameterless2onnx(onnx_path))
        
    
                

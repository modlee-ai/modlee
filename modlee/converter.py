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
from importlib.machinery import SourceFileLoader
import os, inspect
import numpy as np
import torchsummary
import torch
import onnx2torch
import onnx_graphsurgeon as gs
import onnx
import re
from functools import partial
# TODO - cache models for "longer" pipelines
# e.g. torch2code do not need to convert to onnx every time

MODEL_CODE_HEADER = """
import torch, onnx2torch
from torch import tensor
"""

class Converter(object):
    def __init__(self):
        pass

    """
    From Torch
    """
    # def torch2onnx(self, torch_model, input_dummy=torch.randn([10, 3, 300, 300]), tmp_onnx_path="./tmp_model.onnx"):
    def torch_model2onnx_graph(self,
            torch_model,
            input_dummy=torch.randn([10, 3, 300, 300]),
            tmp_onnx_path="./.tmp_model.onnx",
            **kwargs):
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
        input_dummy.requires_grad = False
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
                    # "input_1": {0: "batch_size"},
                    "input_1": [0],
                    # "gemm_1": {0: "batch_size"}
                    "gemm_1": [0],
                },
                **kwargs,
                )
            
            for param in torch_model.parameters():
                param.requires_grad = True
        torch_model.train()
        # input_dummy.requires_grad = True
        # The model we load will have no parameters initialized
        onnx_parameterless_model = onnx.load(tmp_onnx_path)
        # Initialize the parameterless model
        onnx_model = self.onnx_parameterless2onnx(onnx_parameterless_model)
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

    def torch_code2torch_model(self, torch_code: str, tmp_model_path="./.tmp_model.py", *args, **kwargs):
        """
        Convert Torch Code into a Torch Model

        :param torch_code: The Torch Code, either as a file path or the raw code text
        Args:
            torch_code (str): The string representation of the code
            tmp_model_path (str): Where to cache the code as a .py file

        Returns:
            nn.Module: The PyTorch model
        """
        # If the input is a path, load the text from the file
        if os.path.exists(torch_code):
            with open(torch_code, 'r') as _file:
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
            'model_module',
            torch_file).load_module()
        return torch_module.Model()
    code_path2torch = torch_file2torch_model
    

    def torch_model2torch_model(self, torch_model, *args, **kwargs):
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
    torch2torch = torch_model2torch_model
    torch2torch_graph = torch_model2torch_model

    """
    From ONNX
    """

    def onnx_file2torch_model(self, onnx_file, *args, **kwargs):
        """
        Convert an ONNX File to a Torch Model

        :param onnx_file: The ONNX File as a path
        :return torch_model: The Torch Model
        """
        return onnx2torch.convert(onnx_file, *args, **kwargs)
    onnx_path2torch = onnx_file2torch_model
    
    def onnx_file2onnx_graph(self, onnx_file):
        """
        Convert an ONNX File to an ONNX Graph

        :param onnx_file: The ONNX File as a path
        :return onnx_graph: The ONNX Graph as a text
        """
        with open(onnx_file,'r') as _file:
            return self.onnx_text2onnx_graph(_file.read())
    onnx_text_file2onnx = onnx_file2onnx_graph
            
    def onnx_file2torch_model(self, onnx_file):
        """
        Convert an ONNX File to a Torch Model
        
        :param onnx_file: The ONNX File as a path
        :return torch_model: The Torch Model
        """
        # onnx_model = self.onnx_text_file2onnx(onnx_file_path)
        with open(onnx_file,'r') as _file:
            onnx_text = _file.read()
        return self.onnx_text2torch(onnx_text)
    onnx_file2torch = onnx_file2torch_model
        
    def onnx_uninit2torch(self, onnx_graph):
        """
        Convert an uninitialized ONNX Graph to a Torch Model

        :param onnx_graph: The uninitialized ONNX Graph
        :return torch_model: The Torch Model
        """
        return self.onnx_graph2torch_model(
            self.onnx_parameterless2onnx(onnx_graph))
        
    
    def onnx_text2torch_code(self, onnx_text):
        """
        Convert ONNX Text to Torch Code

        :param onnx_text: The ONNX Text
        :return torch_code: The Torch Code
        """
        torch_model = self.onnx_text2torch_model(onnx_text)
        torch_code = self.torch_graph2code(torch_model)
        return torch_code
    onnx_text2code = onnx_text2torch_code 
    # def code2modlee_model(self, code):
    #     torch_model = self.onnx_text2torch(onnx_text_path)
    #     torch_code = self.torch_graph2code(torch_model)
    #     return torch_code        
    # def torch_graph2code(self, torch_graph, *args, **kwargs):
    #     """
    #     Convert a graph-defined PyTorch model to a code representation

    #     Args:
    #         torch_graph (_type_): _description_
    #     """
    #     return self.get_model_code(torch_graph,*args,**kwargs)
            
    def onnx_text2onnx_graph(self, onnx_text):
        """
        Convert ONNX Text to an ONNX Graph

        :param onnx_text: The ONNX Text
        :return onnx_graph: The ONNX Graph
        """
        return onnx.parser.parse_model(onnx_text)
    onnx_text2onnx = onnx_text2onnx_graph
        
    def onnx_text2torch_model(self, onnx_text: bytes):
        """
        Convert ONNX text to Torch model
        """
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
        Convert an ONNX Graph to a Torch Model

        :param onnx_graph: The ONNX Graph object
        :return torch_model: The Torch Model
        """
        return onnx2torch.convert(onnx_graph, *args, **kwargs)
    onnx2torch = onnx_graph2torch_model
        
    def onnx_graph2onnx_text(self, onnx_graph, remove_identity=False):
        """
        Convert an ONNX Graph to ONNX Text

        :param onnx_graph: The ONNX Graph to convert
        :param remove_identity: Whether to remove Identity layers in the output text
        :return onnx_text: The ONNX Text representation
        """

        def get_inner_string(s, _start, _end):
            """
            TODO rewrite the Converter().get_inner_string() to be this simple,
            and handle the string splitting in a wrapper or the methods that use it
            """
            s = s[s.find(_start)+len(_start):]
            s = s[:s.rfind(_end)]
            return s
        
        onnx_str = onnx.printer.to_text(onnx_graph)
        # breakpoint()
        onnx_str = onnx_str.split('\n')
        output_var = "None"
        n_lines = len(onnx_str)
        layer_name_type_dict = {}

        # Regex expression to track floating point number calls" 
        permitted_onxx_float_pattern = r'^(\{)?-?\d*(_\d*)*(e-?\d*)?(\})?[-\d>](,)?$'

        for line_ctr,onnx_uninit_line in enumerate(onnx_str):
            # Skip header
            if line_ctr<6: continue
            
            # Replace characters that cannot be parsed by onnx.parser.parse_model
            unparseable_chars = ['.',':','/']

            # Creating a lsit of string elements to parse
            for unparseable_char in unparseable_chars:
                # Tracking decmimal point separately to enable parsing of floating point numbers                
                # Simply converting other characters to '_' to facilitate parsing.
                onnx_uninit_line = onnx_uninit_line.replace(unparseable_char,'_')
                
            # For NASLib models, handle unparseable characters in e.g. makrograph-edge(7,8)_...
            # Handles the dash, comma, and parentheses
            # TODO -refactor this out into a function
            onnx_uninit_line = re.sub('makrograph-edge\((\d*),(\d*)\)_','makrograph_edge_\\1_\\2_',onnx_uninit_line)
                        
            # Refactor malformed boolean layers
            if 'bool' in onnx_uninit_line:
                onnx_uninit_line = self.refactor_bool_layer(onnx_uninit_line)
                
            # Refactor inf to large value:
            if 'inf' in onnx_uninit_line:
                onnx_uninit_line = self.refactor_inf(onnx_uninit_line)
            # Case: the line is defining a Constant float value that should keep the '.' within brackets {}
            # e.x. const_output_0 = Constant <value = float {0_08}>
            # '0_08' should be reverted back to '0.08'
                
            onnx_uninit_line_as_list = onnx_uninit_line.split(' ')
            for idx,onnx_str_item in enumerate(onnx_uninit_line_as_list):
                # Only replacing decimal point with string if pattern matches, allows capture of floating point values as parameters
                if re.match(permitted_onxx_float_pattern, onnx_str_item):
                    onnx_uninit_line_as_list[idx] = onnx_str_item.replace('_', '.')
                else:
                    continue

            onnx_uninit_line = " ".join(onnx_uninit_line_as_list)
            
            # Found line with output variable, which must be a non-number
            # e.g. "191" is not valid, so we override it with "output_var"
            if "=>" in onnx_uninit_line:
                output_var = get_inner_string(onnx_uninit_line, '=>', '{').strip()
                output_var = get_inner_string(output_var, ']', ')').strip()
                onnx_uninit_line = onnx_uninit_line.replace(f"] {output_var}) {{", f"] output_var) {{")    
            elif line_ctr<(n_lines-1):
                # Add the layer name to the respective layer type in the "counter" dictionary
                layer_name, _, layer_type = onnx_uninit_line.split()[:3]
                if layer_type not in layer_name_type_dict:
                    layer_name_type_dict.update({layer_type:[layer_name]})
                else:
                    layer_name_type_dict[layer_type].append(layer_name)
                                
            onnx_str[line_ctr] = onnx_uninit_line
        
                
        onnx_str = '\n'.join(onnx_str)
        # Replace the output variable with the generic 'output_var'
        onnx_str = onnx_str.replace(f"{output_var} =", f"output_var =")
        
        # Refactor any variables with leading numbers
        onnx_str = self.refactor_leading_number(onnx_str)
        
        
        for layer_type, layer_names in layer_name_type_dict.items():
            for layer_idx,layer_name in enumerate(layer_names):
                if layer_name.isdigit(): continue
                # NOTE - may break if layer names are substrings of other
                # layer names, so pad layer_idx with 0's
                onnx_str = onnx_str.replace(
                    layer_name,
                    f"{layer_type.lower()}_output_{layer_idx:04d}")
                
        if remove_identity:
            onnx_str = self.remove_identity(onnx_str)
        return onnx_str        
    onnx2onnx_text = onnx_graph2onnx_text    
    
    def remove_identity(self, onnx_str):
        """
        Remove identity layers in ONNX text

        :param onnx_str: _description_
        :return: _description_
        """
        
        # Patterns to find 'identity_output_xxxx' assignments and their usage
        #pattern_assignment = re.compile(r'identity_output_(\d{4})\s*=\s*Identity\s*\((onnx__Conv_\d+)\)')
        pattern_assignment = re.compile(r'identity_output_(\d{4})\s*=\s*Identity\s*\(([^)]+)\)')

        assignments = {match[0]: match[1] for match in pattern_assignment.findall(onnx_str)}

        # Remove the assignment lines for 'identity_output_xxxx'
        onnx_str = re.sub(pattern_assignment, '', onnx_str)

        # Replace each instance of 'identity_output_xxxx' with its assigned value
        for identity_number, actual_value in assignments.items():
            onnx_str = onnx_str.replace(f'identity_output_{identity_number}', actual_value)

        # Remove multiple spaces and replace them with a single space
        onnx_str = re.sub(r' +', ' ', onnx_str)
        # Remove chunks of blank space (multiple newlines)
        onnx_str = re.sub(r'\n\s*\n', '\n', onnx_str)

        return onnx_str

    
    def refactor_bool_layer(self, input_str):
        """Refactor boolean layers to the correct number of input elements
        The onnx.printer.to_text() function seems to remove any inputs that the parser would use.
        For example, an int layer is defined like:
        constant_output_0006 = Constant <value = int64[4] {3,12,-1,-1}> ()
          
        From:
        constant_output_0005 = Constant <value = bool[1,1,3,3]___> ()
        To:
        constant_output_0005 = Constant <value = bool[1,1,3,3] {0,0,0,0,0,0,0,0,0}> ()
        
        
        :param input_str: _description_
        """
        if 'bool' not in input_str: return input_str
        bool_dim = re.search('bool\[(?P<bool_dim>.*)\]', input_str)
        if bool_dim:
            bool_dim = bool_dim['bool_dim']
            n_elements = np.prod([int(_b) for _b in bool_dim.split(',')])
            bool_ending = r']'
        else:
            # Handle case where there is no bool dim - should just be one value
            bool_dim = ''
            n_elements = 1
            bool_ending = 'bool'
        input_arg_list = f'{{{",".join("0"*n_elements)}}}'
        # input_str = re.sub('([(bool)\]])(.*)>', f'\\1 {input_arg_list}>', input_str)
        input_str = re.sub(f'{bool_ending}(.*)>', f'{bool_ending} {input_arg_list}>', input_str)
        input_str = re.sub(r'/s{2,}',' ',input_str)
        return input_str
        
    def refactor_inf(self, input_str, large_value='99999999'):
        """Replace 'inf' with a large value because the parser cannot handle infs

        :param input_str: _description_
        
        """
        if 'inf' not in input_str: return input_str
        # return re.sub('float {(-*)inf}',
        #     f'float {{\\1{str(large_value)}}}',input_str)
        return re.sub('inf', str(large_value),input_str)
    
    def refactor_leading_number(self, input_str):
        """Refactor variables with leading numbers which are not parseable,
        0_model_fc_weight -> model_0_fc_weight
        """
        return re.sub('([\s(,]+)(\d+)_*([a-zA-Z0-9]*)[\s_=]','\\1\\3_\\2_',input_str)
    """
    Below are helper functions for importing from torch
    """
    
    def init_graph_tensors(self, graph, tensor_init_fn=partial(np.random.normal,scale=0.01)):
        """
        Initialize the graph's tensors, in place (you do not need to use the return value)
        The input should be an ONNX graph exported from torch without parameters, i.e.
        torch.onnx.export(..., export_params=False)
        This enables exporting to torch
        
        Identity layers get special treatment; they must be initialized for onnx2torch,
        but their target shape is nested within its inputs
        
        Example usage:
        import onnx_graphsurgeon as gs
        graph = gs.import_onnx(path/to/uninitialized/model.onnx)
        Converter().init_graph_tensors(graph)

        Args:
            graph (_type_): _description_
        """
        graph_tensors = graph.tensors()
        for tensor_key,tensor_value in graph_tensors.items():
            # Skip initializing any tensors that are already Constants
            if 'constant' in str(type(tensor_value)).lower():
                if 'identity' not in tensor_key.lower():
                    continue
            # Skip tensors that are inputs/outputs and should be kept as Variables
            # Converting these to constants would essentially "freeze" the network
            # into deterministic outputs
            if any([_substr in tensor_key for _substr in ['input','output']]):
                if 'identity' not in tensor_key.lower():
                    if 'constant' not in tensor_key.lower():
                        continue

            if isinstance(tensor_value, (int,float,)):
                value_shape = (1,)
            elif 'identity' in tensor_key:
                value_shape =  tensor_value.inputs[0].inputs[0].shape
            else:
                value_shape = tensor_value.shape
            if value_shape is None:
                # print(f'{tensor_key} has no value_shape')
                continue
            if isinstance(value_shape[0], str):
                if 'dynamic_axes' in value_shape[0]:
                    # print(f'Skipping {value_shape}')
                    continue
            # print(value_shape, type(value_shape))

            tensor_value.to_constant(
                values=tensor_init_fn(size=value_shape,
                    # dtype=torch.float
                    ).astype(np.float32))
            # if 'identity' in tensor_key: print(tensor_key)
        return graph
    
    
    def init_constant_tensors(graph, constant_tensor_keys: list):
        """
        Given a graph and a list of tensors keys that should be turned constant,
        initialize the tensors with the constant values

        Args:
            graph (_type_): _description_
            constant_tensors (list): _description_
        """
        graph_tensors = graph.tensors()
        for tensor_key in constant_tensor_keys:
            tensor_value = graph_tensors.get(tensor_key, None)
            if tensor_value is None:
                continue
            if isinstance(tensor_value, (int, float,)):
                value_shape = (1,)
            else:
                value_shape = tensor_value.shape
            if value_shape is None:
                continue
            graph_tensors[tensor_key].to_constant(
                values=np.random.uniform(size=value_shape))
        return graph

    # def onnx_parameterless2onnx(self, onnx_path):
    def init_onnx_params(self, onnx_graph):
        """
        Initialize a parameterless ONNX Graph

        :param onnx_graph: The ONNX Graph
        :return onnx_graph: The ONNX Graph with initialized parameters
        """
        # graph = gs.import_onnx(
        #     onnx_model)
        #     # onnx.load(onnx_path))
        # graph = self.init_graph_tensors(graph)
        onnx_graph = self.init_onnx_tensors(onnx_graph)
        return gs.export_onnx(onnx_graph)
    onnx_parameterless2onnx = init_onnx_params
    
    def init_onnx_tensors(self, onnx_graph):
        """
        Initialize the tensors of an ONNX Graph
        TODO - refactor above method with this one
        
        :param onnx_graph: The ONNX Graph
        :return onnx_graph: The ONNX Graph with initalized tensors
        """
        onnx_graph = gs.import_onnx(
            onnx_graph)
        return self.init_graph_tensors(onnx_graph)
    onnx2onnx_gs = init_onnx_tensors
    
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

    def get_inner_string(self, s, _start, _end, return_only_single_value=True):
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
        # If cannot find either substring limiters, return None
        if any([idx==-1 for idx in [s.find(_start),s.find(_end)]]):
            return None
        
        s = s[s.find(_start)+len(_start):]
        s = s[:s.rfind(_end)]
        # valid attributes should just be one string
        if return_only_single_value:
            if len(s.split()) == 1:
                return s
            else:
                return None
        else:
            return s

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
            # print(f"Caught attr {attr_name} on line {line}")
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
        # print(model_attrs)
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
        # breakpoint()
        attr_params = {}

        # if 'initializer' in model_attr: breakpoint()
        for attr_key, attr_val in attr_kwargs.items():
            if attr_key in attrs_to_skip:
                continue
            # if attr_val.default==inspect._empty:
            # if True:
            if hasattr(model_attr, attr_key):
                model_attr_value = getattr(model_attr, attr_key)
                # Convert potentially large tensors to constructors to reduce size
                # if isinstance(model_attr_value, torch.Tensor) or 'tensor' in str(type(model_attr_value)).lower():
                if isinstance(model_attr_value, torch.Tensor):
                    model_attr_value = self.tensor2init(
                        model_attr_value,
                        tensor_type='randn' if 'initial' in attr_key else None)
                attr_params.update({attr_key: model_attr_value})
        if hasattr(model_attr, 'state_dict'):
            # print(f'adding state dict for {model_attr}')
            # breakpoint()
            model_state_dict = model_attr.state_dict()
            # Do this only for initializers because the output file could get large
            model_state_dict = {k:self.tensor2init(v, tensor_type='randn') for k,v in model_state_dict.items() if 'initial' in k}
            # if len(model_state_dict) > 10: breakpoint()
            if len(model_state_dict) > 0:
                attr_params.update(model_state_dict)
                # attr_params.update({'state_dict':model_state_dict})
        # If the model_attr is a torch module that has a state_dict, add it
        return attr_params

    def get_type_string(self, obj) -> str | None:
        """
        Retrieve the type of an object as a string
        TODO - refactor this as a regex

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
        Retrieve the code for a model's __init__() constructor function

        Args:
            model (_type_): The model

        Returns:
            str: An executable __init__ string
        """
        assert model is not None
        model_attrs = self.get_model_attrs_in_forward(model)
        model_attrs.update({onnx_attr:getattr(model,onnx_attr) for onnx_attr in dir(model) if 'onnx' in onnx_attr})
        if hasattr(model, 'initializers'):
            model_attrs.update(
                {'initializers':getattr(model,'initializers')}
            )
        init_attrs = []

        for model_attr_name, model_attr in model_attrs.items():
            attr_params = self.get_params_for_attr(model_attr)
            # if 'initializer' in model_attr_name: breakpoint()
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
            torch.set_printoptions(threshold=np.inf)
            # Convert potentially large tensors to constructors to reduce size
            # kwarg_strs = []
            # for attr_key,attr_value in init_attr[2].items():
            #     if isinstance(attr_value,str):
            #         if 'tensor.' in attr_value.lower():
            kwarg_str = self.kwargs2str(init_attr[2])
        
            # init_line = f"{spacer*2}setattr(self,'{init_attr[0]}', {init_attr[1]}(**{init_attr[2]}))"
            init_line = f"{spacer*2}setattr(self,'{init_attr[0]}', {init_attr[1]}(**{kwarg_str}))"
            init_lines.append(init_line)
            
            # If the attribute is the ONNX initializer,
            # 1) remove the kwargs to the module constructor call
            # 2) append the code to register the initializer states
            if 'initial' in init_attr[0]:
                # remove the just-appended line
                init_line = init_lines.pop().replace('**'+kwarg_str,'')
                init_lines.append(init_line)
                initializer_init_str = self.get_init_module_state_dict_str(f'self.{init_attr[0]}', kwarg_str)
                init_lines.append(initializer_init_str)
        init_code = '\n'.join(init_lines)
        return init_code
    
    def get_init_module_state_dict_str(self, module_name_str:str, state_dict_str:str):
        """Return a string that, when called with exec(), will initialize the torch module's state dictionary.
        The module will likely be a freshly initialized module with an empty state dict.
        This uses register_buffer to add unexpected keys to the state dict.

        :param module_name: The variable name of the module to be initialized, as a string. Must have already been initialized.
        :param state_dict: A string representation of the state_dict
        """
        spacer = "    "
        ret_str = ""
        ret_str += f'{spacer*2}init_state_dict = {state_dict_str}\n'
        ret_str += f'{spacer*2}for k,v in init_state_dict.items():\n'
        ret_str += f'{spacer*3}{module_name_str}.register_buffer(k,v)\n'
        return ret_str
        # return f'{module_nam'
    
    def kwargs2str(self, kwarg_dict):
        """Converts a dictionary of keyword arguments into a properly formatted string
        that can be formatted into an attribute initialization in python code
        """
        # ret_str = ''
        kwarg_list = []
        for kwarg_key,kwarg_value in kwarg_dict.items():
            formatted_kwarg_value = kwarg_value
            if isinstance(formatted_kwarg_value,str):
                try:
                    exec(formatted_kwarg_value)
                except:
                    formatted_kwarg_value = f"\'{formatted_kwarg_value}\'"
            
            kwarg_list.append(f"\'{kwarg_key}\':{formatted_kwarg_value}")
            import os
            with open(os.path.expanduser('~/kwargs.txt'),'a') as _file:
                _file.write(kwarg_list[-1])
                _file.write('\n')
        return f"{{{','.join(kwarg_list)}}}"
        return 
    
    def tensor2init(self, input_tensor, tensor_type:str=None):
        """Converts a monovalue tensor (len(set(tensor))==1) to a string representation of its initialization.
        Converts potentially large from their explicit definition to simply 'tensor.ones((x,y))*values'.
        If `tensor_type` is provided, this function will force-convert a non-uniform tensor to an 
        initialization string for a tensor of that type (e.g. 'randn','ones','zeros').

        :param input_tensor: The tensor to convert
        :param tensor_type: The tensor type to convert to ['randn','zeros','ones']
        """
        if not isinstance(input_tensor, torch.Tensor):
            return input_tensor
        tensor_set = set(input_tensor.flatten().tolist())
        if len(tensor_set) > 1:
            
            torch.set_printoptions(threshold=torch.inf)
            if tensor_type is None:
                # If no type override is defined, get the full class of the tensor
                # to rebuild the tensor into one of the same class
                full_tensor_class = re.search("\'(?P<tensor_class>.*)\'",str(type(input_tensor)))['tensor_class']
                return f'{full_tensor_class}({input_tensor.numpy().tolist()})'
            else:
                # If type override is defined, create a tensor in the shape
                # of the input
                full_tensor_class = f'torch.{tensor_type}'
                return f'{full_tensor_class}({input_tensor.shape})'
        else:
            tensor_value = tensor_set.pop()
            tensor_shape = tuple(input_tensor.shape)
            if abs(tensor_value)<0.00000001:
                output_str = f'torch.zeros({tensor_shape})'
            else:
                output_str = f'torch.ones({tensor_shape})*{tensor_value}'
            return output_str.replace(' ','')

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
            if 'self.Tile' in fwd_line:
                fwd_line = self.convert_tile_layer(fwd_line)
            elif 'self.Gather' in fwd_line:
                fwd_line = self.convert_gather_layer(fwd_line)
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
    torch_graph2code = get_model_code
    
    def convert_tile_layer(self, input_str):
        return re.sub("(Tile.*), (.*)\)", "\\1, list(\\2.type(torch.int64).cpu().numpy()))", input_str)

    def convert_gather_layer(self, input_str):
        return re.sub("(Gather.*), (.*)\)", "\\1, \\2.type(torch.int64))", input_str)
        
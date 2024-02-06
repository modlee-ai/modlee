import pytest, random

import torch, torchvision
import numpy as np
import modlee
from modlee.converter import Converter
converter = Converter()

def test_conversion_pipeline():
    # load a basic resnet model
    torch_model = torchvision.models.resnet18(weights='DEFAULT')    
    # torch model <-> onnx graph
    onnx_graph = converter.torch_model2onnx_graph(torch_model)
    torch_model = converter.onnx_graph2torch_model(onnx_graph)
    
    # onnx graph <-> onnx text
    onnx_text = converter.onnx_graph2onnx_text(onnx_graph)
    onnx_graph = converter.onnx_text2onnx_graph(onnx_text)
    
    # onnx text -> torch code
    torch_code = converter.onnx_text2torch_code(onnx_text)
    
    # torch code -> torch model
    torch_model = converter.torch_code2torch_model(torch_code)
    
    batch_size = random.choice(range(1,33))
    input_dummy = torch.randn((batch_size,3,30,30))
    output_dummy = torch_model(input_dummy)
    assert output_dummy.shape[0] == batch_size
    breakpoint()
    
    # convert from onnx graph to torch model
    # onnx_text = converter.torch_model2onnx_text
    # convert 
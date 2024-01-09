import pytest

import numpy as np
import modlee
from modlee.converter import Converter
converter = Converter()

LEADING_NUMBER_STRS = [
    [' 0_model_fc_weight',' model_0_fc_weight'],
    [',0model_fc_weight',',model_0_fc_weight'],
    ['(0_model_fc_weight','(model_0_fc_weight'],
    [',0_model_fc_bias,', ',model_0_fc_bias,'],
    [',1_model_weight_2',',model_1_weight_2'],
    # [',1_model_weight_1',',model_1_weight_1'],
]
# Zip this and use list as positional argument, as in the following test
@pytest.mark.parametrize('input_output',LEADING_NUMBER_STRS)
def test_refactor_variables_with_leading_numbers(input_output):
    [input_str, actual_output_str] = input_output
    output_str = converter.refactor_leading_number(input_str)
    assert output_str == actual_output_str
    

INPUT_OUTPUT_STRS = [
    ('constant_output_0005 = Constant <value = bool[1,1,3,3]___> ()', 'constant_output_0005 = Constant <value = bool[1,1,3,3] {0,0,0,0,0,0,0,0,0}> ()'),
    ('constant_output_0005 = Constant <value = bool[2,1,1]___> ()', 'constant_output_0005 = Constant <value = bool[2,1,1] {0,0}> ()'),
    ('constant_output_0005 = Constant <value = bool[1]___> ()', 'constant_output_0005 = Constant <value = bool[1] {0}> ()'),
    ('constant_output_0005 = Constant <value = bool ___> ()', 'constant_output_0005 = Constant <value = bool {0}> ()'),
    ('constant_output_0005 = Constant <value = bool___> ()', 'constant_output_0005 = Constant <value = bool {0}> ()'),
    ('constant_output_0005 = Constant <value = bool[1,1,3,3]...> ()', 'constant_output_0005 = Constant <value = bool[1,1,3,3] {0,0,0,0,0,0,0,0,0}> ()'),
    ('constant_output_0005 = Constant <value = bool[2,1,1]...> ()', 'constant_output_0005 = Constant <value = bool[2,1,1] {0,0}> ()'),
    ('constant_output_0005 = Constant <value = bool[1]...> ()', 'constant_output_0005 = Constant <value = bool[1] {0}> ()'),
    ('constant_output_0005 = Constant <value = bool ...> ()', 'constant_output_0005 = Constant <value = bool {0}> ()'),
    ('constant_output_0005 = Constant <value = bool...> ()', 'constant_output_0005 = Constant <value = bool {0}> ()'),
    ('equal_output_0000 = Equal (input_1, constant_output_0001)','equal_output_0000 = Equal (input_1, constant_output_0001)'),
    ('constant_output_0001 = Constant <value = int64 {0}> ()','constant_output_0001 = Constant <value = int64 {0}> ()')
]
@pytest.mark.parametrize('input_str,actual_output_str', INPUT_OUTPUT_STRS)
def test_refactor_bool_layers(input_str, actual_output_str):
    output_str = converter.refactor_bool_layer(input_str)
    assert actual_output_str == output_str
    
INPUT_OUTPUT_KWARGS = [
    ('constant_output_0114 = Constant <value = float {-inf}> ()','constant_output_0114 = Constant <value = float {-99999999}> ()',{}),
    ('constant_output_0017 = Constant <value = float {inf}> ()','constant_output_0017 = Constant <value = float {99999999}> ()',{}),
    ('constant_output_0017 = Constant <value = float {inf}> ()','constant_output_0017 = Constant <value = float {9999}> ()',{'large_value':9999}),
]
@pytest.mark.parametrize('input_str,actual_output_str,fn_kwargs',INPUT_OUTPUT_KWARGS)
def test_refactor_inf(input_str, actual_output_str,fn_kwargs):
    output_str = converter.refactor_inf(input_str, **fn_kwargs)
    assert actual_output_str == output_str
import pytest

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
@pytest.mark.parametrize('input_output',LEADING_NUMBER_STRS)
def test_refactor_variables_with_leading_numbers(input_output):
    [input_str, actual_output_str] = input_output
    output_str = converter.refactor_leading_number(input_str)
    assert output_str == actual_output_str
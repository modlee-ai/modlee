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
    
import torch
torch.set_printoptions(threshold=torch.inf)
random_tensor = torch.randn((5,5))
INPUTS = [
    torch.zeros((3,3)),
    torch.ones((4,4,))*4,
    torch.ones((6,6,))*False,
    torch.ones((6,6,))*True,
    random_tensor,
    # torch.randn(3,3)
]
OUTPUTS = [
    'torch.zeros((3,3))',
    'torch.ones((4,4))*4',
    'torch.zeros((6,6))',
    'torch.ones((6,6))',
    str(random_tensor),
]
@pytest.mark.parametrize('input_value,actual_output_str',zip(INPUTS,OUTPUTS))
def test_large_tensor_to_init(input_value: torch.Tensor, actual_output_str: str):
    fn_output = converter.tensor2init(input_value)
    local_dict = {}
    # breakpoint()
    exec(f'fn_output_array = {fn_output}', globals(), local_dict)
    # breakpoint()
    assert torch.all( (input_value-local_dict['fn_output_array']) < 0.0001 )
    
    # assert torch.all(torch.eq(local_dict['actual_output_array'], local_dict['fn_output_array']))
    pass   


torch.set_printoptions(threshold=torch.inf)
random_tensor = torch.randn((5,5))
INPUTS = [
    random_tensor,
]
OUTPUTS = [
    'torch.randn((5,5))'
]
@pytest.mark.parametrize('input_value,actual_output_str',zip(INPUTS,OUTPUTS))
@pytest.mark.parametrize('tensor_type',['randn','zeros','ones'])
def test_initializer_tensor_to_typed_init(input_value, actual_output_str, tensor_type):
    fn_output = converter.tensor2init(input_value, tensor_type=tensor_type)
    assert tensor_type in fn_output
    local_dict = {}
    exec(f'fn_output_array = {fn_output}', globals(), local_dict)
    assert input_value.shape == local_dict['fn_output_array'].shape
    # random initializations should be different
    assert torch.any( (input_value-local_dict['fn_output_array']) > 0.1 )
    

INPUTS = [
    {'value':'Equal'},
    {'value':'torch.ones((3,3))*3'},
    {'value':'Equal', 'value2':'torch.ones((3,3))*3'},
]
OUTPUTS = [
    "{'value':'Equal'}",
    "{'value':torch.ones((3,3))*3}",
    "{'value':'Equal', 'value2':torch.ones((3,3))*3}",
]
@pytest.mark.parametrize('input_dict,actual_output_str',zip(INPUTS,OUTPUTS))
def test_kwargs_to_str(input_dict, actual_output_str):
    output_str = converter.kwargs2str(input_dict)
    exec(output_str)
    assert actual_output_str.replace(' ','') == output_str.replace(' ','')
    pass

# def test_catch_initializers():
init_state_dict = "{'onnx_initializer_0':torch.randn(torch.Size([32128, 768])),'onnx_initializer_1':torch.randn(torch.Size([768])),'onnx_initializer_2':torch.randn(torch.Size([768, 768])),'onnx_initializer_3':torch.randn(torch.Size([768, 768])),'onnx_initializer_4':torch.randn(torch.Size([768, 768])),'onnx_initializer_5':torch.randn(torch.Size([32, 12])),'onnx_initializer_6':torch.randn(torch.Size([768, 768])),'onnx_initializer_7':torch.randn(torch.Size([768])),'onnx_initializer_8':torch.randn(torch.Size([768, 2048])),'onnx_initializer_9':torch.randn(torch.Size([768, 2048])),'onnx_initializer_10':torch.randn(torch.Size([2048, 768])),'onnx_initializer_11':torch.randn(torch.Size([768])),'onnx_initializer_12':torch.randn(torch.Size([768, 768])),'onnx_initializer_13':torch.randn(torch.Size([768, 768])),'onnx_initializer_14':torch.randn(torch.Size([768, 768])),'onnx_initializer_15':torch.randn(torch.Size([768, 768])),'onnx_initializer_16':torch.randn(torch.Size([768])),'onnx_initializer_17':torch.randn(torch.Size([768, 2048])),'onnx_initializer_18':torch.randn(torch.Size([768, 2048])),'onnx_initializer_19':torch.randn(torch.Size([2048, 768])),'onnx_initializer_20':torch.randn(torch.Size([768])),'onnx_initializer_21':torch.randn(torch.Size([768, 768])),'onnx_initializer_22':torch.randn(torch.Size([768, 768])),'onnx_initializer_23':torch.randn(torch.Size([768, 768])),'onnx_initializer_24':torch.randn(torch.Size([768, 768])),'onnx_initializer_25':torch.randn(torch.Size([768])),'onnx_initializer_26':torch.randn(torch.Size([768, 2048])),'onnx_initializer_27':torch.randn(torch.Size([768, 2048])),'onnx_initializer_28':torch.randn(torch.Size([2048, 768])),'onnx_initializer_29':torch.randn(torch.Size([768])),'onnx_initializer_30':torch.randn(torch.Size([768, 768])),'onnx_initializer_31':torch.randn(torch.Size([768, 768])),'onnx_initializer_32':torch.randn(torch.Size([768, 768])),'onnx_initializer_33':torch.randn(torch.Size([768, 768])),'onnx_initializer_34':torch.randn(torch.Size([768])),'onnx_initializer_35':torch.randn(torch.Size([768, 2048])),'onnx_initializer_36':torch.randn(torch.Size([768, 2048])),'onnx_initializer_37':torch.randn(torch.Size([2048, 768])),'onnx_initializer_38':torch.randn(torch.Size([768])),'onnx_initializer_39':torch.randn(torch.Size([768, 768])),'onnx_initializer_40':torch.randn(torch.Size([768, 768])),'onnx_initializer_41':torch.randn(torch.Size([768, 768])),'onnx_initializer_42':torch.randn(torch.Size([768, 768])),'onnx_initializer_43':torch.randn(torch.Size([768])),'onnx_initializer_44':torch.randn(torch.Size([768, 2048])),'onnx_initializer_45':torch.randn(torch.Size([768, 2048])),'onnx_initializer_46':torch.randn(torch.Size([2048, 768])),'onnx_initializer_47':torch.randn(torch.Size([768])),'onnx_initializer_48':torch.randn(torch.Size([768, 768])),'onnx_initializer_49':torch.randn(torch.Size([768, 768])),'onnx_initializer_50':torch.randn(torch.Size([768, 768])),'onnx_initializer_51':torch.randn(torch.Size([768, 768])),'onnx_initializer_52':torch.randn(torch.Size([768])),'onnx_initializer_53':torch.randn(torch.Size([768, 2048])),'onnx_initializer_54':torch.randn(torch.Size([768, 2048])),'onnx_initializer_55':torch.randn(torch.Size([2048, 768])),'onnx_initializer_56':torch.randn(torch.Size([768])),'onnx_initializer_57':torch.randn(torch.Size([768, 768])),'onnx_initializer_58':torch.randn(torch.Size([768, 768])),'onnx_initializer_59':torch.randn(torch.Size([768, 768])),'onnx_initializer_60':torch.randn(torch.Size([768, 768])),'onnx_initializer_61':torch.randn(torch.Size([768])),'onnx_initializer_62':torch.randn(torch.Size([768, 2048])),'onnx_initializer_63':torch.randn(torch.Size([768, 2048])),'onnx_initializer_64':torch.randn(torch.Size([2048, 768])),'onnx_initializer_65':torch.randn(torch.Size([768])),'onnx_initializer_66':torch.randn(torch.Size([768, 768])),'onnx_initializer_67':torch.randn(torch.Size([768, 768])),'onnx_initializer_68':torch.randn(torch.Size([768, 768])),'onnx_initializer_69':torch.randn(torch.Size([768, 768])),'onnx_initializer_70':torch.randn(torch.Size([768])),'onnx_initializer_71':torch.randn(torch.Size([768, 2048])),'onnx_initializer_72':torch.randn(torch.Size([768, 2048])),'onnx_initializer_73':torch.randn(torch.Size([2048, 768])),'onnx_initializer_74':torch.randn(torch.Size([768])),'onnx_initializer_75':torch.randn(torch.Size([768, 768])),'onnx_initializer_76':torch.randn(torch.Size([768, 768])),'onnx_initializer_77':torch.randn(torch.Size([768, 768])),'onnx_initializer_78':torch.randn(torch.Size([768, 768])),'onnx_initializer_79':torch.randn(torch.Size([768])),'onnx_initializer_80':torch.randn(torch.Size([768, 2048])),'onnx_initializer_81':torch.randn(torch.Size([768, 2048])),'onnx_initializer_82':torch.randn(torch.Size([2048, 768])),'onnx_initializer_83':torch.randn(torch.Size([768])),'onnx_initializer_84':torch.randn(torch.Size([768, 768])),'onnx_initializer_85':torch.randn(torch.Size([768, 768])),'onnx_initializer_86':torch.randn(torch.Size([768, 768])),'onnx_initializer_87':torch.randn(torch.Size([768, 768])),'onnx_initializer_88':torch.randn(torch.Size([768])),'onnx_initializer_89':torch.randn(torch.Size([768, 2048])),'onnx_initializer_90':torch.randn(torch.Size([768, 2048])),'onnx_initializer_91':torch.randn(torch.Size([2048, 768])),'onnx_initializer_92':torch.randn(torch.Size([768])),'onnx_initializer_93':torch.randn(torch.Size([768, 768])),'onnx_initializer_94':torch.randn(torch.Size([768, 768])),'onnx_initializer_95':torch.randn(torch.Size([768, 768])),'onnx_initializer_96':torch.randn(torch.Size([768, 768])),'onnx_initializer_97':torch.randn(torch.Size([768])),'onnx_initializer_98':torch.randn(torch.Size([768, 2048])),'onnx_initializer_99':torch.randn(torch.Size([768, 2048])),'onnx_initializer_100':torch.randn(torch.Size([2048, 768])),'onnx_initializer_101':torch.randn(torch.Size([768])),'onnx_initializer_102':torch.randn(torch.Size([768, 768])),'onnx_initializer_103':torch.randn(torch.Size([768, 768])),'onnx_initializer_104':torch.randn(torch.Size([768, 768])),'onnx_initializer_105':torch.randn(torch.Size([768, 768])),'onnx_initializer_106':torch.randn(torch.Size([768])),'onnx_initializer_107':torch.randn(torch.Size([768, 2048])),'onnx_initializer_108':torch.randn(torch.Size([768, 2048])),'onnx_initializer_109':torch.randn(torch.Size([2048, 768])),'onnx_initializer_110':torch.randn(torch.Size([768]))}"
def test_init_initializer():
    import re
    exec(f'actual_state_dict = {init_state_dict}', globals(), locals())
    # state_keys = re.findall("\'([a-zA-Z0-9_]*)\'",init_state_dict)
    # state_values = re.findall("(torch[a-zA-Z0-9_\[\]\(\)\.,]*)[}\']",init_state_dict.replace(' ',''))
    init_tensor = torch.nn.modules.module.Module()
    output_str = converter.get_init_module_state_dict_str('init_tensor', init_state_dict)
    exec(f'{output_str}', globals(), locals())
    init_tensor = locals()['init_tensor']
    # output_tensor = locals()['output_tensor']
    for k,v in locals()['actual_state_dict'].items():
        assert hasattr(init_tensor, k)
        assert v.shape == getattr(init_tensor,k).shape
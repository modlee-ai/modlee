
import torch, onnx2torch
from torch import tensor

class Model(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        setattr(self,'Identity', onnx2torch.node_converters.identity.OnnxCopyIdentity(**{}))
        setattr(self,'Identity_1', onnx2torch.node_converters.identity.OnnxCopyIdentity(**{}))
        setattr(self,'Identity_2', onnx2torch.node_converters.identity.OnnxCopyIdentity(**{}))
        setattr(self,'Identity_3', onnx2torch.node_converters.identity.OnnxCopyIdentity(**{}))
        setattr(self,'Identity_4', onnx2torch.node_converters.identity.OnnxCopyIdentity(**{}))
        setattr(self,'Identity_5', onnx2torch.node_converters.identity.OnnxCopyIdentity(**{}))
        setattr(self,'Shape', onnx2torch.node_converters.shape.OnnxShape(**{'start':0,'end':None}))
        setattr(self,'Constant', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.ones(())*2}))
        setattr(self,'Gather', onnx2torch.node_converters.gather.OnnxGather(**{'axis':0}))
        setattr(self,'Shape_1', onnx2torch.node_converters.shape.OnnxShape(**{'start':0,'end':None}))
        setattr(self,'Constant_1', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.ones(())*3}))
        setattr(self,'Gather_1', onnx2torch.node_converters.gather.OnnxGather(**{'axis':0}))
        setattr(self,'Conv', torch.nn.modules.conv.Conv2d(**{'in_channels':3,'out_channels':64,'kernel_size':(7, 7),'stride':(2, 2),'padding':(3, 3),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'MaxPool', torch.nn.modules.pooling.MaxPool2d(**{'kernel_size':[3, 3],'stride':[2, 2],'padding':[1, 1],'dilation':[1, 1],'return_indices':False,'ceil_mode':False}))
        setattr(self,'Conv_1', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':64,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_1', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_2', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':64,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_2', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_3', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Conv_4', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_3', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_5', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':64,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_4', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_6', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':64,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_5', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_7', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_1', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_6', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_8', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':64,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_7', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_9', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':64,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_8', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_10', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_2', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_9', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_11', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':128,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_10', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_12', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':128,'kernel_size':(3, 3),'stride':(2, 2),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_11', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_13', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':512,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Conv_14', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':512,'kernel_size':(1, 1),'stride':(2, 2),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_3', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_12', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_15', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':128,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_13', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_16', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':128,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_14', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_17', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':512,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_4', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_15', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_18', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':128,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_16', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_19', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':128,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_17', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_20', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':512,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_5', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_18', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_21', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':128,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_19', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_22', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':128,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_20', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_23', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':512,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_6', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_21', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_24', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_22', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_25', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_23', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_26', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':1024,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Conv_27', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':1024,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_7', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_24', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_28', torch.nn.modules.conv.Conv2d(**{'in_channels':1024,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_25', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_29', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(2, 2),'dilation':(2, 2),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_26', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_30', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':1024,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_8', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_27', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_31', torch.nn.modules.conv.Conv2d(**{'in_channels':1024,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_28', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_32', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(2, 2),'dilation':(2, 2),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_29', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_33', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':1024,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_9', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_30', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_34', torch.nn.modules.conv.Conv2d(**{'in_channels':1024,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_31', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_35', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(2, 2),'dilation':(2, 2),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_32', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_36', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':1024,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_10', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_33', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_37', torch.nn.modules.conv.Conv2d(**{'in_channels':1024,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_34', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_38', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(2, 2),'dilation':(2, 2),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_35', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_39', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':1024,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_11', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_36', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_40', torch.nn.modules.conv.Conv2d(**{'in_channels':1024,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_37', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_41', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(2, 2),'dilation':(2, 2),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_38', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_42', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':1024,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_12', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_39', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_43', torch.nn.modules.conv.Conv2d(**{'in_channels':1024,'out_channels':512,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_40', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_44', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':512,'kernel_size':(3, 3),'stride':(1, 1),'padding':(2, 2),'dilation':(2, 2),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_41', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_45', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':2048,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Conv_46', torch.nn.modules.conv.Conv2d(**{'in_channels':1024,'out_channels':2048,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_13', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_42', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_47', torch.nn.modules.conv.Conv2d(**{'in_channels':2048,'out_channels':512,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_43', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_48', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':512,'kernel_size':(3, 3),'stride':(1, 1),'padding':(4, 4),'dilation':(4, 4),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_44', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_49', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':2048,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_14', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_45', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_50', torch.nn.modules.conv.Conv2d(**{'in_channels':2048,'out_channels':512,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_46', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_51', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':512,'kernel_size':(3, 3),'stride':(1, 1),'padding':(4, 4),'dilation':(4, 4),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_47', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_52', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':2048,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_15', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_48', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_53', torch.nn.modules.conv.Conv2d(**{'in_channels':2048,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_49', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_54', torch.nn.modules.conv.Conv2d(**{'in_channels':2048,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(12, 12),'dilation':(12, 12),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_50', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_55', torch.nn.modules.conv.Conv2d(**{'in_channels':2048,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(24, 24),'dilation':(24, 24),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_51', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_56', torch.nn.modules.conv.Conv2d(**{'in_channels':2048,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(36, 36),'dilation':(36, 36),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_52', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Shape_2', onnx2torch.node_converters.shape.OnnxShape(**{'start':0,'end':None}))
        setattr(self,'Constant_2', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.ones(())*2}))
        setattr(self,'Gather_2', onnx2torch.node_converters.gather.OnnxGather(**{'axis':0}))
        setattr(self,'Shape_3', onnx2torch.node_converters.shape.OnnxShape(**{'start':0,'end':None}))
        setattr(self,'Constant_3', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.ones(())*3}))
        setattr(self,'Gather_3', onnx2torch.node_converters.gather.OnnxGather(**{'axis':0}))
        setattr(self,'GlobalAveragePool', onnx2torch.node_converters.global_average_pool.OnnxGlobalAveragePoolWithKnownInputShape(**{'input_shape':[0, 2048, 38, 38]}))
        setattr(self,'Conv_57', torch.nn.modules.conv.Conv2d(**{'in_channels':2048,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_53', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Constant_4', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
        setattr(self,'Unsqueeze', onnx2torch.node_converters.unsqueeze.OnnxUnsqueezeStaticAxes(**{'axes':[0]}))
        setattr(self,'Constant_5', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
        setattr(self,'Unsqueeze_1', onnx2torch.node_converters.unsqueeze.OnnxUnsqueezeStaticAxes(**{'axes':[0]}))
        setattr(self,'Concat', onnx2torch.node_converters.concat.OnnxConcat(**{'axis':0}))
        setattr(self,'Shape_4', onnx2torch.node_converters.shape.OnnxShape(**{'start':0,'end':None}))
        setattr(self,'Constant_6', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
        setattr(self,'Constant_7', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
        setattr(self,'Constant_8', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.ones((1,))*2}))
        setattr(self,'Slice', onnx2torch.node_converters.slice.OnnxSlice(**{}))
        setattr(self,'Cast', onnx2torch.node_converters.cast.OnnxCast(**{'onnx_dtype':7}))
        setattr(self,'Concat_1', onnx2torch.node_converters.concat.OnnxConcat(**{'axis':0}))
        setattr(self,'Resize', onnx2torch.node_converters.resize.OnnxResize(**{'mode':'linear','align_corners':False,'ignore_roi':True,'ignore_bs_ch_size':False}))
        setattr(self,'Concat_2', onnx2torch.node_converters.concat.OnnxConcat(**{'axis':1}))
        setattr(self,'Conv_58', torch.nn.modules.conv.Conv2d(**{'in_channels':1280,'out_channels':256,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_54', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_59', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_55', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_60', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':10,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Constant_9', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
        setattr(self,'Unsqueeze_2', onnx2torch.node_converters.unsqueeze.OnnxUnsqueezeStaticAxes(**{'axes':[0]}))
        setattr(self,'Constant_10', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
        setattr(self,'Unsqueeze_3', onnx2torch.node_converters.unsqueeze.OnnxUnsqueezeStaticAxes(**{'axes':[0]}))
        setattr(self,'Concat_3', onnx2torch.node_converters.concat.OnnxConcat(**{'axis':0}))
        setattr(self,'Shape_5', onnx2torch.node_converters.shape.OnnxShape(**{'start':0,'end':None}))
        setattr(self,'Constant_11', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
        setattr(self,'Constant_12', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
        setattr(self,'Constant_13', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.ones((1,))*2}))
        setattr(self,'Slice_1', onnx2torch.node_converters.slice.OnnxSlice(**{}))
        setattr(self,'Cast_1', onnx2torch.node_converters.cast.OnnxCast(**{'onnx_dtype':7}))
        setattr(self,'Concat_4', onnx2torch.node_converters.concat.OnnxConcat(**{'axis':0}))
        setattr(self,'Resize_1', onnx2torch.node_converters.resize.OnnxResize(**{'mode':'linear','align_corners':False,'ignore_roi':True,'ignore_bs_ch_size':False}))
        setattr(self,'initializers', torch.nn.modules.module.Module())
        init_state_dict = {'onnx_initializer_0':torch.randn(torch.Size([256])),'onnx_initializer_1':torch.randn(torch.Size([256])),'onnx_initializer_2':torch.randn(torch.Size([256])),'onnx_initializer_3':torch.randn(torch.Size([256])),'onnx_initializer_4':torch.randn(torch.Size([256])),'onnx_initializer_5':torch.randn(torch.Size([256]))}
        for k,v in init_state_dict.items():
            self.initializers.register_buffer(k,v)


    def forward(self, input_1):
        initializers_onnx_initializer_0 = self.initializers.onnx_initializer_0
        identity = self.Identity(initializers_onnx_initializer_0);  initializers_onnx_initializer_0 = None
        initializers_onnx_initializer_1 = self.initializers.onnx_initializer_1
        identity_1 = self.Identity_1(initializers_onnx_initializer_1);  initializers_onnx_initializer_1 = None
        initializers_onnx_initializer_2 = self.initializers.onnx_initializer_2
        identity_2 = self.Identity_2(initializers_onnx_initializer_2);  initializers_onnx_initializer_2 = None
        initializers_onnx_initializer_3 = self.initializers.onnx_initializer_3
        identity_3 = self.Identity_3(initializers_onnx_initializer_3);  initializers_onnx_initializer_3 = None
        initializers_onnx_initializer_4 = self.initializers.onnx_initializer_4
        identity_4 = self.Identity_4(initializers_onnx_initializer_4);  initializers_onnx_initializer_4 = None
        initializers_onnx_initializer_5 = self.initializers.onnx_initializer_5
        identity_5 = self.Identity_5(initializers_onnx_initializer_5);  initializers_onnx_initializer_5 = None
        shape = self.Shape(input_1)
        constant = self.Constant()
        gather = self.Gather(shape, constant.type(torch.int64));  shape = constant = None
        shape_1 = self.Shape_1(input_1)
        constant_1 = self.Constant_1()
        gather_1 = self.Gather_1(shape_1, constant_1.type(torch.int64));  shape_1 = constant_1 = None
        conv = self.Conv(input_1);  input_1 = None
        relu = self.Relu(conv);  conv = None
        max_pool = self.MaxPool(relu);  relu = None
        conv_1 = self.Conv_1(max_pool)
        relu_1 = self.Relu_1(conv_1);  conv_1 = None
        conv_2 = self.Conv_2(relu_1);  relu_1 = None
        relu_2 = self.Relu_2(conv_2);  conv_2 = None
        conv_3 = self.Conv_3(relu_2);  relu_2 = None
        conv_4 = self.Conv_4(max_pool);  max_pool = None
        add = self.Add(conv_3, conv_4);  conv_3 = conv_4 = None
        relu_3 = self.Relu_3(add);  add = None
        conv_5 = self.Conv_5(relu_3)
        relu_4 = self.Relu_4(conv_5);  conv_5 = None
        conv_6 = self.Conv_6(relu_4);  relu_4 = None
        relu_5 = self.Relu_5(conv_6);  conv_6 = None
        conv_7 = self.Conv_7(relu_5);  relu_5 = None
        add_1 = self.Add_1(conv_7, relu_3);  conv_7 = relu_3 = None
        relu_6 = self.Relu_6(add_1);  add_1 = None
        conv_8 = self.Conv_8(relu_6)
        relu_7 = self.Relu_7(conv_8);  conv_8 = None
        conv_9 = self.Conv_9(relu_7);  relu_7 = None
        relu_8 = self.Relu_8(conv_9);  conv_9 = None
        conv_10 = self.Conv_10(relu_8);  relu_8 = None
        add_2 = self.Add_2(conv_10, relu_6);  conv_10 = relu_6 = None
        relu_9 = self.Relu_9(add_2);  add_2 = None
        conv_11 = self.Conv_11(relu_9)
        relu_10 = self.Relu_10(conv_11);  conv_11 = None
        conv_12 = self.Conv_12(relu_10);  relu_10 = None
        relu_11 = self.Relu_11(conv_12);  conv_12 = None
        conv_13 = self.Conv_13(relu_11);  relu_11 = None
        conv_14 = self.Conv_14(relu_9);  relu_9 = None
        add_3 = self.Add_3(conv_13, conv_14);  conv_13 = conv_14 = None
        relu_12 = self.Relu_12(add_3);  add_3 = None
        conv_15 = self.Conv_15(relu_12)
        relu_13 = self.Relu_13(conv_15);  conv_15 = None
        conv_16 = self.Conv_16(relu_13);  relu_13 = None
        relu_14 = self.Relu_14(conv_16);  conv_16 = None
        conv_17 = self.Conv_17(relu_14);  relu_14 = None
        add_4 = self.Add_4(conv_17, relu_12);  conv_17 = relu_12 = None
        relu_15 = self.Relu_15(add_4);  add_4 = None
        conv_18 = self.Conv_18(relu_15)
        relu_16 = self.Relu_16(conv_18);  conv_18 = None
        conv_19 = self.Conv_19(relu_16);  relu_16 = None
        relu_17 = self.Relu_17(conv_19);  conv_19 = None
        conv_20 = self.Conv_20(relu_17);  relu_17 = None
        add_5 = self.Add_5(conv_20, relu_15);  conv_20 = relu_15 = None
        relu_18 = self.Relu_18(add_5);  add_5 = None
        conv_21 = self.Conv_21(relu_18)
        relu_19 = self.Relu_19(conv_21);  conv_21 = None
        conv_22 = self.Conv_22(relu_19);  relu_19 = None
        relu_20 = self.Relu_20(conv_22);  conv_22 = None
        conv_23 = self.Conv_23(relu_20);  relu_20 = None
        add_6 = self.Add_6(conv_23, relu_18);  conv_23 = relu_18 = None
        relu_21 = self.Relu_21(add_6);  add_6 = None
        conv_24 = self.Conv_24(relu_21)
        relu_22 = self.Relu_22(conv_24);  conv_24 = None
        conv_25 = self.Conv_25(relu_22);  relu_22 = None
        relu_23 = self.Relu_23(conv_25);  conv_25 = None
        conv_26 = self.Conv_26(relu_23);  relu_23 = None
        conv_27 = self.Conv_27(relu_21);  relu_21 = None
        add_7 = self.Add_7(conv_26, conv_27);  conv_26 = conv_27 = None
        relu_24 = self.Relu_24(add_7);  add_7 = None
        conv_28 = self.Conv_28(relu_24)
        relu_25 = self.Relu_25(conv_28);  conv_28 = None
        conv_29 = self.Conv_29(relu_25);  relu_25 = None
        relu_26 = self.Relu_26(conv_29);  conv_29 = None
        conv_30 = self.Conv_30(relu_26);  relu_26 = None
        add_8 = self.Add_8(conv_30, relu_24);  conv_30 = relu_24 = None
        relu_27 = self.Relu_27(add_8);  add_8 = None
        conv_31 = self.Conv_31(relu_27)
        relu_28 = self.Relu_28(conv_31);  conv_31 = None
        conv_32 = self.Conv_32(relu_28);  relu_28 = None
        relu_29 = self.Relu_29(conv_32);  conv_32 = None
        conv_33 = self.Conv_33(relu_29);  relu_29 = None
        add_9 = self.Add_9(conv_33, relu_27);  conv_33 = relu_27 = None
        relu_30 = self.Relu_30(add_9);  add_9 = None
        conv_34 = self.Conv_34(relu_30)
        relu_31 = self.Relu_31(conv_34);  conv_34 = None
        conv_35 = self.Conv_35(relu_31);  relu_31 = None
        relu_32 = self.Relu_32(conv_35);  conv_35 = None
        conv_36 = self.Conv_36(relu_32);  relu_32 = None
        add_10 = self.Add_10(conv_36, relu_30);  conv_36 = relu_30 = None
        relu_33 = self.Relu_33(add_10);  add_10 = None
        conv_37 = self.Conv_37(relu_33)
        relu_34 = self.Relu_34(conv_37);  conv_37 = None
        conv_38 = self.Conv_38(relu_34);  relu_34 = None
        relu_35 = self.Relu_35(conv_38);  conv_38 = None
        conv_39 = self.Conv_39(relu_35);  relu_35 = None
        add_11 = self.Add_11(conv_39, relu_33);  conv_39 = relu_33 = None
        relu_36 = self.Relu_36(add_11);  add_11 = None
        conv_40 = self.Conv_40(relu_36)
        relu_37 = self.Relu_37(conv_40);  conv_40 = None
        conv_41 = self.Conv_41(relu_37);  relu_37 = None
        relu_38 = self.Relu_38(conv_41);  conv_41 = None
        conv_42 = self.Conv_42(relu_38);  relu_38 = None
        add_12 = self.Add_12(conv_42, relu_36);  conv_42 = relu_36 = None
        relu_39 = self.Relu_39(add_12);  add_12 = None
        conv_43 = self.Conv_43(relu_39)
        relu_40 = self.Relu_40(conv_43);  conv_43 = None
        conv_44 = self.Conv_44(relu_40);  relu_40 = None
        relu_41 = self.Relu_41(conv_44);  conv_44 = None
        conv_45 = self.Conv_45(relu_41);  relu_41 = None
        conv_46 = self.Conv_46(relu_39);  relu_39 = None
        add_13 = self.Add_13(conv_45, conv_46);  conv_45 = conv_46 = None
        relu_42 = self.Relu_42(add_13);  add_13 = None
        conv_47 = self.Conv_47(relu_42)
        relu_43 = self.Relu_43(conv_47);  conv_47 = None
        conv_48 = self.Conv_48(relu_43);  relu_43 = None
        relu_44 = self.Relu_44(conv_48);  conv_48 = None
        conv_49 = self.Conv_49(relu_44);  relu_44 = None
        add_14 = self.Add_14(conv_49, relu_42);  conv_49 = relu_42 = None
        relu_45 = self.Relu_45(add_14);  add_14 = None
        conv_50 = self.Conv_50(relu_45)
        relu_46 = self.Relu_46(conv_50);  conv_50 = None
        conv_51 = self.Conv_51(relu_46);  relu_46 = None
        relu_47 = self.Relu_47(conv_51);  conv_51 = None
        conv_52 = self.Conv_52(relu_47);  relu_47 = None
        add_15 = self.Add_15(conv_52, relu_45);  conv_52 = relu_45 = None
        relu_48 = self.Relu_48(add_15);  add_15 = None
        conv_53 = self.Conv_53(relu_48)
        relu_49 = self.Relu_49(conv_53);  conv_53 = None
        conv_54 = self.Conv_54(relu_48)
        relu_50 = self.Relu_50(conv_54);  conv_54 = None
        conv_55 = self.Conv_55(relu_48)
        relu_51 = self.Relu_51(conv_55);  conv_55 = None
        conv_56 = self.Conv_56(relu_48)
        relu_52 = self.Relu_52(conv_56);  conv_56 = None
        shape_2 = self.Shape_2(relu_48)
        constant_2 = self.Constant_2()
        gather_2 = self.Gather_2(shape_2, constant_2.type(torch.int64));  shape_2 = constant_2 = None
        shape_3 = self.Shape_3(relu_48)
        constant_3 = self.Constant_3()
        gather_3 = self.Gather_3(shape_3, constant_3.type(torch.int64));  shape_3 = constant_3 = None
        global_average_pool = self.GlobalAveragePool(relu_48);  relu_48 = None
        conv_57 = self.Conv_57(global_average_pool);  global_average_pool = None
        relu_53 = self.Relu_53(conv_57);  conv_57 = None
        constant_4 = self.Constant_4()
        unsqueeze = self.Unsqueeze(gather_2);  gather_2 = None
        constant_5 = self.Constant_5()
        unsqueeze_1 = self.Unsqueeze_1(gather_3);  gather_3 = None
        concat = self.Concat(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        shape_4 = self.Shape_4(relu_53)
        constant_6 = self.Constant_6()
        constant_7 = self.Constant_7()
        constant_8 = self.Constant_8()
        slice_1 = self.Slice(shape_4, constant_7, constant_8, constant_6);  shape_4 = constant_7 = constant_8 = constant_6 = None
        cast = self.Cast(concat);  concat = None
        concat_1 = self.Concat_1(slice_1, cast);  slice_1 = cast = None
        resize = self.Resize(relu_53, sizes = concat_1);  relu_53 = concat_1 = None
        concat_2 = self.Concat_2(relu_49, relu_50, relu_51, relu_52, resize);  relu_49 = relu_50 = relu_51 = relu_52 = resize = None
        conv_58 = self.Conv_58(concat_2);  concat_2 = None
        relu_54 = self.Relu_54(conv_58);  conv_58 = None
        conv_59 = self.Conv_59(relu_54);  relu_54 = None
        relu_55 = self.Relu_55(conv_59);  conv_59 = None
        conv_60 = self.Conv_60(relu_55);  relu_55 = None
        constant_9 = self.Constant_9()
        unsqueeze_2 = self.Unsqueeze_2(gather);  gather = None
        constant_10 = self.Constant_10()
        unsqueeze_3 = self.Unsqueeze_3(gather_1);  gather_1 = None
        concat_3 = self.Concat_3(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
        shape_5 = self.Shape_5(conv_60)
        constant_11 = self.Constant_11()
        constant_12 = self.Constant_12()
        constant_13 = self.Constant_13()
        slice_2 = self.Slice_1(shape_5, constant_12, constant_13, constant_11);  shape_5 = constant_12 = constant_13 = constant_11 = None
        cast_1 = self.Cast_1(concat_3);  concat_3 = None
        concat_4 = self.Concat_4(slice_2, cast_1);  slice_2 = cast_1 = None
        resize_1 = self.Resize_1(conv_60, sizes = concat_4);  conv_60 = concat_4 = None
        return resize_1
    

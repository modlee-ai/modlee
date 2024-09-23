
import torch, onnx2torch
from torch import tensor
class Model(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        setattr(self,'Conv', torch.nn.modules.conv.Conv2d(**{'in_channels':3,'out_channels':3,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Conv_1', torch.nn.modules.conv.Conv2d(**{'in_channels':3,'out_channels':64,'kernel_size':(7, 7),'stride':(2, 2),'padding':(3, 3),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'MaxPool', torch.nn.modules.pooling.MaxPool2d(**{'kernel_size':[3, 3],'stride':[2, 2],'padding':[1, 1],'dilation':[1, 1],'return_indices':False,'ceil_mode':False}))
        setattr(self,'Conv_2', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':64,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_1', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_3', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':64,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_2', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_4', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':64,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_3', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_5', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':64,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_1', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_4', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_6', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':128,'kernel_size':(3, 3),'stride':(2, 2),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_5', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_7', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':128,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Conv_8', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':128,'kernel_size':(1, 1),'stride':(2, 2),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_2', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_6', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_9', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':128,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_7', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_10', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':128,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_3', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_8', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_11', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':256,'kernel_size':(3, 3),'stride':(2, 2),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_9', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_12', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Conv_13', torch.nn.modules.conv.Conv2d(**{'in_channels':128,'out_channels':256,'kernel_size':(1, 1),'stride':(2, 2),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_4', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_10', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_14', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_11', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_15', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':256,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_5', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_12', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_16', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':512,'kernel_size':(3, 3),'stride':(2, 2),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_13', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_17', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':512,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Conv_18', torch.nn.modules.conv.Conv2d(**{'in_channels':256,'out_channels':512,'kernel_size':(1, 1),'stride':(2, 2),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_6', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_14', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_19', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':512,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Relu_15', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'Conv_20', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':512,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
        setattr(self,'Add_7', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
        setattr(self,'Relu_16', torch.nn.modules.activation.ReLU(**{'inplace':False}))
        setattr(self,'GlobalAveragePool', onnx2torch.node_converters.global_average_pool.OnnxGlobalAveragePoolWithKnownInputShape(**{'input_shape':[0, 512, 1, 1]}))
        setattr(self,'Flatten', torch.nn.modules.flatten.Flatten(**{'start_dim':1,'end_dim':-1}))
        setattr(self,'Gemm', torch.nn.modules.linear.Linear(**{'in_features':512,'out_features':1000}))
        setattr(self,'Gemm_1', torch.nn.modules.linear.Linear(**{'in_features':1000,'out_features':3}))
    def forward(self, input_1):
        conv = self.Conv(input_1);  input_1 = None
        conv_1 = self.Conv_1(conv);  conv = None
        relu = self.Relu(conv_1);  conv_1 = None
        max_pool = self.MaxPool(relu);  relu = None
        conv_2 = self.Conv_2(max_pool)
        relu_1 = self.Relu_1(conv_2);  conv_2 = None
        conv_3 = self.Conv_3(relu_1);  relu_1 = None
        add = self.Add(conv_3, max_pool);  conv_3 = max_pool = None
        relu_2 = self.Relu_2(add);  add = None
        conv_4 = self.Conv_4(relu_2)
        relu_3 = self.Relu_3(conv_4);  conv_4 = None
        conv_5 = self.Conv_5(relu_3);  relu_3 = None
        add_1 = self.Add_1(conv_5, relu_2);  conv_5 = relu_2 = None
        relu_4 = self.Relu_4(add_1);  add_1 = None
        conv_6 = self.Conv_6(relu_4)
        relu_5 = self.Relu_5(conv_6);  conv_6 = None
        conv_7 = self.Conv_7(relu_5);  relu_5 = None
        conv_8 = self.Conv_8(relu_4);  relu_4 = None
        add_2 = self.Add_2(conv_7, conv_8);  conv_7 = conv_8 = None
        relu_6 = self.Relu_6(add_2);  add_2 = None
        conv_9 = self.Conv_9(relu_6)
        relu_7 = self.Relu_7(conv_9);  conv_9 = None
        conv_10 = self.Conv_10(relu_7);  relu_7 = None
        add_3 = self.Add_3(conv_10, relu_6);  conv_10 = relu_6 = None
        relu_8 = self.Relu_8(add_3);  add_3 = None
        conv_11 = self.Conv_11(relu_8)
        relu_9 = self.Relu_9(conv_11);  conv_11 = None
        conv_12 = self.Conv_12(relu_9);  relu_9 = None
        conv_13 = self.Conv_13(relu_8);  relu_8 = None
        add_4 = self.Add_4(conv_12, conv_13);  conv_12 = conv_13 = None
        relu_10 = self.Relu_10(add_4);  add_4 = None
        conv_14 = self.Conv_14(relu_10)
        relu_11 = self.Relu_11(conv_14);  conv_14 = None
        conv_15 = self.Conv_15(relu_11);  relu_11 = None
        add_5 = self.Add_5(conv_15, relu_10);  conv_15 = relu_10 = None
        relu_12 = self.Relu_12(add_5);  add_5 = None
        conv_16 = self.Conv_16(relu_12)
        relu_13 = self.Relu_13(conv_16);  conv_16 = None
        conv_17 = self.Conv_17(relu_13);  relu_13 = None
        conv_18 = self.Conv_18(relu_12);  relu_12 = None
        add_6 = self.Add_6(conv_17, conv_18);  conv_17 = conv_18 = None
        relu_14 = self.Relu_14(add_6);  add_6 = None
        conv_19 = self.Conv_19(relu_14)
        relu_15 = self.Relu_15(conv_19);  conv_19 = None
        conv_20 = self.Conv_20(relu_15);  relu_15 = None
        add_7 = self.Add_7(conv_20, relu_14);  conv_20 = relu_14 = None
        relu_16 = self.Relu_16(add_7);  add_7 = None
        global_average_pool = self.GlobalAveragePool(relu_16);  relu_16 = None
        flatten = self.Flatten(global_average_pool);  global_average_pool = None
        gemm = self.Gemm(flatten);  flatten = None
        gemm_1 = self.Gemm_1(gemm);  gemm = None
        return gemm_1
    

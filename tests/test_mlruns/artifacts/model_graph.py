import torch, onnx2torch
from torch import tensor


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        setattr(
            self,
            "Conv",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 3,
                    "out_channels": 64,
                    "kernel_size": (7, 7),
                    "stride": (2, 2),
                    "padding": (3, 3),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "MaxPool",
            torch.nn.modules.pooling.MaxPool2d(
                **{
                    "kernel_size": [3, 3],
                    "stride": [2, 2],
                    "padding": [1, 1],
                    "dilation": 1,
                    "return_indices": False,
                    "ceil_mode": False,
                }
            ),
        )
        setattr(
            self,
            "Conv_1",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 64,
                    "out_channels": 64,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_1", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_2",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 64,
                    "out_channels": 64,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_2", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_3",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 64,
                    "out_channels": 128,
                    "kernel_size": (3, 3),
                    "stride": (2, 2),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_3", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_4",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 128,
                    "out_channels": 128,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Conv_5",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 64,
                    "out_channels": 128,
                    "kernel_size": (1, 1),
                    "stride": (2, 2),
                    "padding": (0, 0),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_1",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_4", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_6",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 128,
                    "out_channels": 128,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_5", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_7",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 128,
                    "out_channels": 128,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_2",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_6", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_8",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 128,
                    "out_channels": 128,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_7", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_9",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 128,
                    "out_channels": 128,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_3",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_8", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_10",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 128,
                    "out_channels": 128,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_9", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_11",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 128,
                    "out_channels": 128,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_4",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_10", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_12",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 128,
                    "out_channels": 256,
                    "kernel_size": (3, 3),
                    "stride": (2, 2),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_11", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_13",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 256,
                    "out_channels": 256,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Conv_14",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 128,
                    "out_channels": 256,
                    "kernel_size": (1, 1),
                    "stride": (2, 2),
                    "padding": (0, 0),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_5",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_12", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_15",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 256,
                    "out_channels": 256,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_13", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_16",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 256,
                    "out_channels": 256,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_6",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_14", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_17",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 256,
                    "out_channels": 256,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_15", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_18",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 256,
                    "out_channels": 256,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_7",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_16", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_19",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 256,
                    "out_channels": 256,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_17", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_20",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 256,
                    "out_channels": 256,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_8",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_18", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_21",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 256,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (2, 2),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_19", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_22",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 512,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Conv_23",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 256,
                    "out_channels": 512,
                    "kernel_size": (1, 1),
                    "stride": (2, 2),
                    "padding": (0, 0),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_9",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_20", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_24",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 512,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_21", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_25",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 512,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_10",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_22", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_26",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 512,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_23", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_27",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 512,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_11",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_24", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_28",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 512,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_25", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_29",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 512,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_12",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_26", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_30",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 512,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_27", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_31",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 512,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_13",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_28", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_32",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 512,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(self, "Relu_29", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "Conv_33",
            torch.nn.modules.conv.Conv2d(
                **{
                    "in_channels": 512,
                    "out_channels": 512,
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "padding_mode": "zeros",
                }
            ),
        )
        setattr(
            self,
            "Add_14",
            onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(
                **{"operation_type": "Add", "broadcast": None, "axis": None}
            ),
        )
        setattr(self, "Relu_30", torch.nn.modules.activation.ReLU(**{"inplace": False}))
        setattr(
            self,
            "GlobalAveragePool",
            onnx2torch.node_converters.global_average_pool.OnnxGlobalAveragePoolWithKnownInputShape(
                **{"input_shape": [8, 512, 1, 1]}
            ),
        )
        setattr(
            self,
            "Flatten",
            torch.nn.modules.flatten.Flatten(**{"start_dim": 1, "end_dim": -1}),
        )
        setattr(
            self,
            "Gemm",
            torch.nn.modules.linear.Linear(
                **{"in_features": 512, "out_features": 1000}
            ),
        )

    def forward(self, input_1):
        conv = self.Conv(input_1)
        input_1 = None
        relu = self.Relu(conv)
        conv = None
        max_pool = self.MaxPool(relu)
        relu = None
        conv_1 = self.Conv_1(max_pool)
        relu_1 = self.Relu_1(conv_1)
        conv_1 = None
        conv_2 = self.Conv_2(relu_1)
        relu_1 = None
        add = self.Add(conv_2, max_pool)
        conv_2 = max_pool = None
        relu_2 = self.Relu_2(add)
        add = None
        conv_3 = self.Conv_3(relu_2)
        relu_3 = self.Relu_3(conv_3)
        conv_3 = None
        conv_4 = self.Conv_4(relu_3)
        relu_3 = None
        conv_5 = self.Conv_5(relu_2)
        relu_2 = None
        add_1 = self.Add_1(conv_4, conv_5)
        conv_4 = conv_5 = None
        relu_4 = self.Relu_4(add_1)
        add_1 = None
        conv_6 = self.Conv_6(relu_4)
        relu_5 = self.Relu_5(conv_6)
        conv_6 = None
        conv_7 = self.Conv_7(relu_5)
        relu_5 = None
        add_2 = self.Add_2(conv_7, relu_4)
        conv_7 = relu_4 = None
        relu_6 = self.Relu_6(add_2)
        add_2 = None
        conv_8 = self.Conv_8(relu_6)
        relu_7 = self.Relu_7(conv_8)
        conv_8 = None
        conv_9 = self.Conv_9(relu_7)
        relu_7 = None
        add_3 = self.Add_3(conv_9, relu_6)
        conv_9 = relu_6 = None
        relu_8 = self.Relu_8(add_3)
        add_3 = None
        conv_10 = self.Conv_10(relu_8)
        relu_9 = self.Relu_9(conv_10)
        conv_10 = None
        conv_11 = self.Conv_11(relu_9)
        relu_9 = None
        add_4 = self.Add_4(conv_11, relu_8)
        conv_11 = relu_8 = None
        relu_10 = self.Relu_10(add_4)
        add_4 = None
        conv_12 = self.Conv_12(relu_10)
        relu_11 = self.Relu_11(conv_12)
        conv_12 = None
        conv_13 = self.Conv_13(relu_11)
        relu_11 = None
        conv_14 = self.Conv_14(relu_10)
        relu_10 = None
        add_5 = self.Add_5(conv_13, conv_14)
        conv_13 = conv_14 = None
        relu_12 = self.Relu_12(add_5)
        add_5 = None
        conv_15 = self.Conv_15(relu_12)
        relu_13 = self.Relu_13(conv_15)
        conv_15 = None
        conv_16 = self.Conv_16(relu_13)
        relu_13 = None
        add_6 = self.Add_6(conv_16, relu_12)
        conv_16 = relu_12 = None
        relu_14 = self.Relu_14(add_6)
        add_6 = None
        conv_17 = self.Conv_17(relu_14)
        relu_15 = self.Relu_15(conv_17)
        conv_17 = None
        conv_18 = self.Conv_18(relu_15)
        relu_15 = None
        add_7 = self.Add_7(conv_18, relu_14)
        conv_18 = relu_14 = None
        relu_16 = self.Relu_16(add_7)
        add_7 = None
        conv_19 = self.Conv_19(relu_16)
        relu_17 = self.Relu_17(conv_19)
        conv_19 = None
        conv_20 = self.Conv_20(relu_17)
        relu_17 = None
        add_8 = self.Add_8(conv_20, relu_16)
        conv_20 = relu_16 = None
        relu_18 = self.Relu_18(add_8)
        add_8 = None
        conv_21 = self.Conv_21(relu_18)
        relu_19 = self.Relu_19(conv_21)
        conv_21 = None
        conv_22 = self.Conv_22(relu_19)
        relu_19 = None
        conv_23 = self.Conv_23(relu_18)
        relu_18 = None
        add_9 = self.Add_9(conv_22, conv_23)
        conv_22 = conv_23 = None
        relu_20 = self.Relu_20(add_9)
        add_9 = None
        conv_24 = self.Conv_24(relu_20)
        relu_21 = self.Relu_21(conv_24)
        conv_24 = None
        conv_25 = self.Conv_25(relu_21)
        relu_21 = None
        add_10 = self.Add_10(conv_25, relu_20)
        conv_25 = relu_20 = None
        relu_22 = self.Relu_22(add_10)
        add_10 = None
        conv_26 = self.Conv_26(relu_22)
        relu_23 = self.Relu_23(conv_26)
        conv_26 = None
        conv_27 = self.Conv_27(relu_23)
        relu_23 = None
        add_11 = self.Add_11(conv_27, relu_22)
        conv_27 = relu_22 = None
        relu_24 = self.Relu_24(add_11)
        add_11 = None
        conv_28 = self.Conv_28(relu_24)
        relu_25 = self.Relu_25(conv_28)
        conv_28 = None
        conv_29 = self.Conv_29(relu_25)
        relu_25 = None
        add_12 = self.Add_12(conv_29, relu_24)
        conv_29 = relu_24 = None
        relu_26 = self.Relu_26(add_12)
        add_12 = None
        conv_30 = self.Conv_30(relu_26)
        relu_27 = self.Relu_27(conv_30)
        conv_30 = None
        conv_31 = self.Conv_31(relu_27)
        relu_27 = None
        add_13 = self.Add_13(conv_31, relu_26)
        conv_31 = relu_26 = None
        relu_28 = self.Relu_28(add_13)
        add_13 = None
        conv_32 = self.Conv_32(relu_28)
        relu_29 = self.Relu_29(conv_32)
        conv_32 = None
        conv_33 = self.Conv_33(relu_29)
        relu_29 = None
        add_14 = self.Add_14(conv_33, relu_28)
        conv_33 = relu_28 = None
        relu_30 = self.Relu_30(add_14)
        add_14 = None
        global_average_pool = self.GlobalAveragePool(relu_30)
        relu_30 = None
        flatten = self.Flatten(global_average_pool)
        global_average_pool = None
        gemm = self.Gemm(flatten)
        flatten = None
        return gemm

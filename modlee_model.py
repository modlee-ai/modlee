
main_graph (float[10,3,300,300] input_1, float[10,256,1,1] model_classifier_model_4_weight, float[10] model_classifier_model_4_bias, float[64,3,7,7] onnx__Conv_594, float[64] onnx__Conv_595, float[64,64,1,1] onnx__Conv_597, float[64] onnx__Conv_598, float[64,64,3,3] onnx__Conv_600, float[64] onnx__Conv_601, float[256,64,1,1] onnx__Conv_603, float[256] onnx__Conv_604, float[256,64,1,1] onnx__Conv_606, float[256] onnx__Conv_607, float[64,256,1,1] onnx__Conv_609, float[64] onnx__Conv_610, float[64,64,3,3] onnx__Conv_612, float[64] onnx__Conv_613, float[256,64,1,1] onnx__Conv_615, float[256] onnx__Conv_616, float[64,256,1,1] onnx__Conv_618, float[64] onnx__Conv_619, float[64,64,3,3] onnx__Conv_621, float[64] onnx__Conv_622, float[256,64,1,1] onnx__Conv_624, float[256] onnx__Conv_625, float[128,256,1,1] onnx__Conv_627, float[128] onnx__Conv_628, float[128,128,3,3] onnx__Conv_630, float[128] onnx__Conv_631, float[512,128,1,1] onnx__Conv_633, float[512] onnx__Conv_634, float[512,256,1,1] onnx__Conv_636, float[512] onnx__Conv_637, float[128,512,1,1] onnx__Conv_639, float[128] onnx__Conv_640, float[128,128,3,3] onnx__Conv_642, float[128] onnx__Conv_643, float[512,128,1,1] onnx__Conv_645, float[512] onnx__Conv_646, float[128,512,1,1] onnx__Conv_648, float[128] onnx__Conv_649, float[128,128,3,3] onnx__Conv_651, float[128] onnx__Conv_652, float[512,128,1,1] onnx__Conv_654, float[512] onnx__Conv_655, float[128,512,1,1] onnx__Conv_657, float[128] onnx__Conv_658, float[128,128,3,3] onnx__Conv_660, float[128] onnx__Conv_661, float[512,128,1,1] onnx__Conv_663, float[512] onnx__Conv_664, float[256,512,1,1] onnx__Conv_666, float[256] onnx__Conv_667, float[256,256,3,3] onnx__Conv_669, float[256] onnx__Conv_670, float[1024,256,1,1] onnx__Conv_672, float[1024] onnx__Conv_673, float[1024,512,1,1] onnx__Conv_675, float[1024] onnx__Conv_676, float[256,1024,1,1] onnx__Conv_678, float[256] onnx__Conv_679, float[256,256,3,3] onnx__Conv_681, float[256] onnx__Conv_682, float[1024,256,1,1] onnx__Conv_684, float[1024] onnx__Conv_685, float[256,1024,1,1] onnx__Conv_687, float[256] onnx__Conv_688, float[256,256,3,3] onnx__Conv_690, float[256] onnx__Conv_691, float[1024,256,1,1] onnx__Conv_693, float[1024] onnx__Conv_694, float[256,1024,1,1] onnx__Conv_696, float[256] onnx__Conv_697, float[256,256,3,3] onnx__Conv_699, float[256] onnx__Conv_700, float[1024,256,1,1] onnx__Conv_702, float[1024] onnx__Conv_703, float[256,1024,1,1] onnx__Conv_705, float[256] onnx__Conv_706, float[256,256,3,3] onnx__Conv_708, float[256] onnx__Conv_709, float[1024,256,1,1] onnx__Conv_711, float[1024] onnx__Conv_712, float[256,1024,1,1] onnx__Conv_714, float[256] onnx__Conv_715, float[256,256,3,3] onnx__Conv_717, float[256] onnx__Conv_718, float[1024,256,1,1] onnx__Conv_720, float[1024] onnx__Conv_721, float[512,1024,1,1] onnx__Conv_723, float[512] onnx__Conv_724, float[512,512,3,3] onnx__Conv_726, float[512] onnx__Conv_727, float[2048,512,1,1] onnx__Conv_729, float[2048] onnx__Conv_730, float[2048,1024,1,1] onnx__Conv_732, float[2048] onnx__Conv_733, float[512,2048,1,1] onnx__Conv_735, float[512] onnx__Conv_736, float[512,512,3,3] onnx__Conv_738, float[512] onnx__Conv_739, float[2048,512,1,1] onnx__Conv_741, float[2048] onnx__Conv_742, float[512,2048,1,1] onnx__Conv_744, float[512] onnx__Conv_745, float[512,512,3,3] onnx__Conv_747, float[512] onnx__Conv_748, float[2048,512,1,1] onnx__Conv_750, float[2048] onnx__Conv_751, float[256,2048,1,1] onnx__Conv_753, float[256] onnx__Conv_754, float[256,2048,3,3] onnx__Conv_756, float[256,2048,3,3] onnx__Conv_759, float[256,2048,3,3] onnx__Conv_762, float[256,2048,1,1] onnx__Conv_765, float[256,1280,1,1] onnx__Conv_768, float[256,256,3,3] onnx__Conv_771) => (float[10,10,300,300] output_var) {
   identity_output_0000 = Identity (onnx__Conv_754)
   identity_output_0001 = Identity (onnx__Conv_754)
   identity_output_0002 = Identity (onnx__Conv_754)
   identity_output_0003 = Identity (onnx__Conv_754)
   identity_output_0004 = Identity (onnx__Conv_754)
   identity_output_0005 = Identity (onnx__Conv_754)
   conv_output_0000 = Conv <dilations = [1, 1], group = 1, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]> (input_1, onnx__Conv_594, onnx__Conv_595)
   relu_output_0000 = Relu (conv_output_0000)
   maxpool_output_0000 = MaxPool <ceil_mode = 0, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]> (relu_output_0000)
   conv_output_0001 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (maxpool_output_0000, onnx__Conv_597, onnx__Conv_598)
   relu_output_0001 = Relu (conv_output_0001)
   conv_output_0002 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0001, onnx__Conv_600, onnx__Conv_601)
   relu_output_0002 = Relu (conv_output_0002)
   conv_output_0003 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0002, onnx__Conv_603, onnx__Conv_604)
   conv_output_0004 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (maxpool_output_0000, onnx__Conv_606, onnx__Conv_607)
   add_output_0000 = Add (conv_output_0003, conv_output_0004)
   relu_output_0003 = Relu (add_output_0000)
   conv_output_0005 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0003, onnx__Conv_609, onnx__Conv_610)
   relu_output_0004 = Relu (conv_output_0005)
   conv_output_0006 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0004, onnx__Conv_612, onnx__Conv_613)
   relu_output_0005 = Relu (conv_output_0006)
   conv_output_0007 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0005, onnx__Conv_615, onnx__Conv_616)
   add_output_0001 = Add (conv_output_0007, relu_output_0003)
   relu_output_0006 = Relu (add_output_0001)
   conv_output_0008 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0006, onnx__Conv_618, onnx__Conv_619)
   relu_output_0007 = Relu (conv_output_0008)
   conv_output_0009 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0007, onnx__Conv_621, onnx__Conv_622)
   relu_output_0008 = Relu (conv_output_0009)
   conv_output_0010 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0008, onnx__Conv_624, onnx__Conv_625)
   add_output_0002 = Add (conv_output_0010, relu_output_0006)
   relu_output_0009 = Relu (add_output_0002)
   conv_output_0011 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0009, onnx__Conv_627, onnx__Conv_628)
   relu_output_0010 = Relu (conv_output_0011)
   conv_output_0012 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]> (relu_output_0010, onnx__Conv_630, onnx__Conv_631)
   relu_output_0011 = Relu (conv_output_0012)
   conv_output_0013 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0011, onnx__Conv_633, onnx__Conv_634)
   conv_output_0014 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [2, 2]> (relu_output_0009, onnx__Conv_636, onnx__Conv_637)
   add_output_0003 = Add (conv_output_0013, conv_output_0014)
   relu_output_0012 = Relu (add_output_0003)
   conv_output_0015 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0012, onnx__Conv_639, onnx__Conv_640)
   relu_output_0013 = Relu (conv_output_0015)
   conv_output_0016 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0013, onnx__Conv_642, onnx__Conv_643)
   relu_output_0014 = Relu (conv_output_0016)
   conv_output_0017 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0014, onnx__Conv_645, onnx__Conv_646)
   add_output_0004 = Add (conv_output_0017, relu_output_0012)
   relu_output_0015 = Relu (add_output_0004)
   conv_output_0018 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0015, onnx__Conv_648, onnx__Conv_649)
   relu_output_0016 = Relu (conv_output_0018)
   conv_output_0019 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0016, onnx__Conv_651, onnx__Conv_652)
   relu_output_0017 = Relu (conv_output_0019)
   conv_output_0020 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0017, onnx__Conv_654, onnx__Conv_655)
   add_output_0005 = Add (conv_output_0020, relu_output_0015)
   relu_output_0018 = Relu (add_output_0005)
   conv_output_0021 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0018, onnx__Conv_657, onnx__Conv_658)
   relu_output_0019 = Relu (conv_output_0021)
   conv_output_0022 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0019, onnx__Conv_660, onnx__Conv_661)
   relu_output_0020 = Relu (conv_output_0022)
   conv_output_0023 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0020, onnx__Conv_663, onnx__Conv_664)
   add_output_0006 = Add (conv_output_0023, relu_output_0018)
   relu_output_0021 = Relu (add_output_0006)
   conv_output_0024 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0021, onnx__Conv_666, onnx__Conv_667)
   relu_output_0022 = Relu (conv_output_0024)
   conv_output_0025 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0022, onnx__Conv_669, onnx__Conv_670)
   relu_output_0023 = Relu (conv_output_0025)
   conv_output_0026 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0023, onnx__Conv_672, onnx__Conv_673)
   conv_output_0027 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0021, onnx__Conv_675, onnx__Conv_676)
   add_output_0007 = Add (conv_output_0026, conv_output_0027)
   relu_output_0024 = Relu (add_output_0007)
   conv_output_0028 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0024, onnx__Conv_678, onnx__Conv_679)
   relu_output_0025 = Relu (conv_output_0028)
   conv_output_0029 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0025, onnx__Conv_681, onnx__Conv_682)
   relu_output_0026 = Relu (conv_output_0029)
   conv_output_0030 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0026, onnx__Conv_684, onnx__Conv_685)
   add_output_0008 = Add (conv_output_0030, relu_output_0024)
   relu_output_0027 = Relu (add_output_0008)
   conv_output_0031 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0027, onnx__Conv_687, onnx__Conv_688)
   relu_output_0028 = Relu (conv_output_0031)
   conv_output_0032 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0028, onnx__Conv_690, onnx__Conv_691)
   relu_output_0029 = Relu (conv_output_0032)
   conv_output_0033 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0029, onnx__Conv_693, onnx__Conv_694)
   add_output_0009 = Add (conv_output_0033, relu_output_0027)
   relu_output_0030 = Relu (add_output_0009)
   conv_output_0034 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0030, onnx__Conv_696, onnx__Conv_697)
   relu_output_0031 = Relu (conv_output_0034)
   conv_output_0035 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0031, onnx__Conv_699, onnx__Conv_700)
   relu_output_0032 = Relu (conv_output_0035)
   conv_output_0036 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0032, onnx__Conv_702, onnx__Conv_703)
   add_output_0010 = Add (conv_output_0036, relu_output_0030)
   relu_output_0033 = Relu (add_output_0010)
   conv_output_0037 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0033, onnx__Conv_705, onnx__Conv_706)
   relu_output_0034 = Relu (conv_output_0037)
   conv_output_0038 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0034, onnx__Conv_708, onnx__Conv_709)
   relu_output_0035 = Relu (conv_output_0038)
   conv_output_0039 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0035, onnx__Conv_711, onnx__Conv_712)
   add_output_0011 = Add (conv_output_0039, relu_output_0033)
   relu_output_0036 = Relu (add_output_0011)
   conv_output_0040 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0036, onnx__Conv_714, onnx__Conv_715)
   relu_output_0037 = Relu (conv_output_0040)
   conv_output_0041 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0037, onnx__Conv_717, onnx__Conv_718)
   relu_output_0038 = Relu (conv_output_0041)
   conv_output_0042 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0038, onnx__Conv_720, onnx__Conv_721)
   add_output_0012 = Add (conv_output_0042, relu_output_0036)
   relu_output_0039 = Relu (add_output_0012)
   conv_output_0043 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0039, onnx__Conv_723, onnx__Conv_724)
   relu_output_0040 = Relu (conv_output_0043)
   conv_output_0044 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0040, onnx__Conv_726, onnx__Conv_727)
   relu_output_0041 = Relu (conv_output_0044)
   conv_output_0045 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0041, onnx__Conv_729, onnx__Conv_730)
   conv_output_0046 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0039, onnx__Conv_732, onnx__Conv_733)
   add_output_0013 = Add (conv_output_0045, conv_output_0046)
   relu_output_0042 = Relu (add_output_0013)
   conv_output_0047 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0042, onnx__Conv_735, onnx__Conv_736)
   relu_output_0043 = Relu (conv_output_0047)
   conv_output_0048 = Conv <dilations = [4, 4], group = 1, kernel_shape = [3, 3], pads = [4, 4, 4, 4], strides = [1, 1]> (relu_output_0043, onnx__Conv_738, onnx__Conv_739)
   relu_output_0044 = Relu (conv_output_0048)
   conv_output_0049 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0044, onnx__Conv_741, onnx__Conv_742)
   add_output_0014 = Add (conv_output_0049, relu_output_0042)
   relu_output_0045 = Relu (add_output_0014)
   conv_output_0050 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0045, onnx__Conv_744, onnx__Conv_745)
   relu_output_0046 = Relu (conv_output_0050)
   conv_output_0051 = Conv <dilations = [4, 4], group = 1, kernel_shape = [3, 3], pads = [4, 4, 4, 4], strides = [1, 1]> (relu_output_0046, onnx__Conv_747, onnx__Conv_748)
   relu_output_0047 = Relu (conv_output_0051)
   conv_output_0052 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0047, onnx__Conv_750, onnx__Conv_751)
   add_output_0015 = Add (conv_output_0052, relu_output_0045)
   relu_output_0048 = Relu (add_output_0015)
   conv_output_0053 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0048, onnx__Conv_753, onnx__Conv_754)
   relu_output_0049 = Relu (conv_output_0053)
   conv_output_0054 = Conv <dilations = [12, 12], group = 1, kernel_shape = [3, 3], pads = [12, 12, 12, 12], strides = [1, 1]> (relu_output_0048, onnx__Conv_756, identity_output_0005)
   relu_output_0050 = Relu (conv_output_0054)
   conv_output_0055 = Conv <dilations = [24, 24], group = 1, kernel_shape = [3, 3], pads = [24, 24, 24, 24], strides = [1, 1]> (relu_output_0048, onnx__Conv_759, identity_output_0004)
   relu_output_0051 = Relu (conv_output_0055)
   conv_output_0056 = Conv <dilations = [36, 36], group = 1, kernel_shape = [3, 3], pads = [36, 36, 36, 36], strides = [1, 1]> (relu_output_0048, onnx__Conv_762, identity_output_0003)
   relu_output_0052 = Relu (conv_output_0056)
   globalaveragepool_output_0000 = GlobalAveragePool (relu_output_0048)
   conv_output_0057 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (globalaveragepool_output_0000, onnx__Conv_765, identity_output_0002)
   relu_output_0053 = Relu (conv_output_0057)
   shape_output_0000 = Shape (relu_output_0053)
   constant_output_0000 = Constant <value = int64[1] {0}> ()
   constant_output_0001 = Constant <value = int64[1] {0}> ()
   constant_output_0002 = Constant <value = int64[1] {2}> ()
   slice_output_0000 = Slice (shape_output_0000, constant_output_0001, constant_output_0002, constant_output_0000)
   constant_output_0003 = Constant <value = int64[2] {38,38}> ()
   concat_output_0000 = Concat <axis = 0> (slice_output_0000, constant_output_0003)
   resize_output_0000 = Resize <coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -0.75, mode = "linear", nearest_mode = "floor"> (relu_output_0053, , , concat_output_0000)
   concat_output_0001 = Concat <axis = 1> (relu_output_0049, relu_output_0050, relu_output_0051, relu_output_0052, resize_output_0000)
   conv_output_0058 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (concat_output_0001, onnx__Conv_768, identity_output_0001)
   relu_output_0054 = Relu (conv_output_0058)
   conv_output_0059 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0054, onnx__Conv_771, identity_output_0000)
   relu_output_0055 = Relu (conv_output_0059)
   conv_output_0060 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0055, model_classifier_model_4_weight, model_classifier_model_4_bias)
   shape_output_0001 = Shape (conv_output_0060)
   constant_output_0004 = Constant <value = int64[1] {0}> ()
   constant_output_0005 = Constant <value = int64[1] {0}> ()
   constant_output_0006 = Constant <value = int64[1] {2}> ()
   slice_output_0001 = Slice (shape_output_0001, constant_output_0005, constant_output_0006, constant_output_0004)
   constant_output_0007 = Constant <value = int64[2] {300,300}> ()
   concat_output_0002 = Concat <axis = 0> (slice_output_0001, constant_output_0007)
   output_var = Resize <coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -0.75, mode = "linear", nearest_mode = "floor"> (conv_output_0060, , , concat_output_0002)
}
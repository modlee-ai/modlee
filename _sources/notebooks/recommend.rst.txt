Recommendation
==============

Why use Modlee?
---------------

   At Modlee, we’re on a mission to ensure that everyone, everywhere has
   access to top-tier machine learning solutions. We’re flipping the
   script on how ML knowledge is shared, going beyond the realms of
   Hugging Face, GitHub, and Papers with Code. Let’s be honest, we’re
   all diving into similar models, right? Modlee is your turbocharged
   ticket to effortlessly and swiftly connect with the ideal models for
   your datasets, making your journey smoother, faster, and with minimal
   effort on your part.

[image]

We’re working towards this vision, and would love give you a sneak peak
of our technology. Some of the below features in this demo are at
different stages of development.

Here’s how it works in Pytorch
------------------------------

1) You prepare your dataset:

::

           training_dataloader = torch.utils.data.DataLoader(...)

2) Modlee recommends a model close to your target solution by analyzing
   your dataset and solution requirements:

::

           modlee_model = modlee.Recommender(training_dataloader,max_model_size_MB=10, ...)

3) While you train the model, Modlee prepares everything you need for
   your convenience:

::

           modlee_model.train()

4) Modlee auto-documents your experiment locally and learns from
   non-sensitive details to enhance ML model recommendations for the
   community:

::

           modlee_model.train_documentation_locations()

Let’s see what Modlee recommends for MNIST … (~5 mins)
------------------------------------------------------

First let’s quickly install the modlee package, should take ~10 seconds.
Thanks for your patience!

.. code:: ipython3

    import os
    SERVER_ENDPOINT = 'http://ec2-3-84-155-233.compute-1.amazonaws.com:7070'
    
    def setup(demo_header='demos_demo04_'):
        os.system(
            f'curl -o modlee-0.0.1.post6-py3-none-any.whl {SERVER_ENDPOINT}/get_wheel/{demo_header}modlee-0.0.1.post6-py3-none-any.whl -O')
        os.system(
            f'curl -o modleesurvey-0.0.1-py3-none-any.whl {SERVER_ENDPOINT}/get_wheel/{demo_header}modleesurvey-0.0.1-py3-none-any.whl -O  > /dev/null 2>&1')
        os.system(
            f'curl -o onnx2torch-1.5.11-py3-none-any.whl {SERVER_ENDPOINT}/get_wheel/{demo_header}onnx2torch-1.5.11-py3-none-any.whl -O  > /dev/null 2>&1')
        os.system("pip3 install -q 'modlee-0.0.1.post6-py3-none-any.whl' 'modleesurvey-0.0.1-py3-none-any.whl' 'onnx2torch-1.5.11-py3-none-any.whl' torch==2.1.0 torchsummary==1.5.1 ipywidgets==7.7.1  > /dev/null 2>&1")
        # os.system("pip3 install -q 'modleesurvey-0.0.1-py3-none-any.whl' 'onnx2torch-1.5.11-py3-none-any.whl' torchsummary==1.5.1 ipywidgets  > /dev/null 2>&1")
        os.system("pip3 install -q onnx_graphsurgeon==0.3.27 --index-url https://pypi.ngc.nvidia.com  > /dev/null 2>&1")
    setup()
      
    import modlee
    modlee.init(api_key="community",run_dir='./')


.. parsed-literal::

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 39751  100 39751    0     0  7763k      0 --:--:-- --:--:-- --:--:-- 9704k


1. You prepare your dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import torch, torchvision
    import torchvision.transforms as transforms
    from torchvision.transforms import v2
    # torch.set_default_device('cuda')
    
    transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.Resize((300,300))])
    def remap_255(x, n_unique=21):
        # return x
        mask = x!=255
        # mask = mask.to('cuda')
        # mask = mask.to(x.)
        x = x.type(torch.LongTensor)
        # x = x.to('cuda')
        mask = mask.to(x.device)
        # print(x.device, mask.device)
        x = x.where(mask, n_unique-1)
        x = x.squeeze()
        return x
        return x.type(torch.LongTensor).where(mask, n_unique-1).squeeze().to('cuda')
        
    transforms = v2.Compose(
        [
            # v2.ToImage(),
            # v2.RandomPhotometricDistort(p=1),
            # v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
            # v2.RandomIoUCrop(),
            # v2.RandomHorizontalFlip(p=1),
            # v2.SanitizeBoundingBoxes(),
            v2.ToTensor(),
            v2.Resize((300,300)),
            # v2.ToDtype(torch.float32, scale=True),
            # v2.ToTensor(),
            # v2.Lambda(remap_255)
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
    train_dataset = torchvision.datasets.VOCSegmentation(
        root='./data', year='2007',
        image_set='test',
        # image_set='train',
        download=True,
        transform=transforms,
        target_transform=v2.Compose([v2.ToTensor(), v2.Resize((300,300)), v2.Lambda(remap_255)])
        )
    train_dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(train_dataset, )
    
    # train_dataset.data.to(torch.device('cuda'))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=64,
        batch_size=16,
        pin_memory=True,
        # num_workers=torch.cuda.device_count()*4
        # collate_fn=lambda batch: list(zip(*batch))
        )
    # train_dataloader.to(torch.device('cuda'))


.. parsed-literal::

    /home/ubuntu/projects/.venv/lib/python3.10/site-packages/torchvision/transforms/v2/_deprecated.py:43: UserWarning: The transform `ToTensor()` is deprecated and will be removed in a future release. Instead, please use `v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])`.
      warnings.warn(


.. parsed-literal::

    Files already downloaded and verified
    Using downloaded and verified file: ./data/VOCtest_06-Nov-2007.tar
    Extracting ./data/VOCtest_06-Nov-2007.tar to ./data


2. Modlee recommends a model close to your target solution by analyzing your dataset and solution requirements:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    recommender = modlee.recommender.from_modality_task(
        modality='image',
        # task='classification',
        task='segmentation',
        )
    recommender.fit(train_dataloader)
    modlee_model = recommender.model 



.. parsed-literal::

    [Modlee] -> Just a moment, analyzing your dataset ...
    


.. code:: ipython3

    recommender.get_model_details()


.. parsed-literal::

    --- Modlee Recommended Model Details --->
    
    [Modlee] -> In case you want to take a deeper look, I saved the summary of my current model recommendation here:
                        file: ./modlee_model.txt
    
    [Modlee] -> I also saved the model as a python editable version (model def, train, val, optimizer):
                        file: ./modlee_model.py
                This is a great place to start your own model exploration!


.. code:: ipython3

    !cat ./modlee_model.txt
    !cat ./modlee_model.py
    # train_dataloader.dataset.to('cuda')
    b1,b2 = next(iter(train_dataloader))
    print(b1.device)
    # modlee_model.to(torch.device('cuda'))
    print(modlee_model.device, b1.device)
    # b1.to(modlee_model.device)
    # modlee_model(b1.to(modlee_model.device)).shape
    modlee_model(b1).shape


.. parsed-literal::

    <
       ir_version: 9,
       opset_import: ["" : 17],
       producer_name: "pytorch",
       producer_version: "2.2.0"
    >
    main_graph (float[input_1_dynamic_axes_1,3,300,300] input_1, float[21,512,1,1] model_classifier_model_4_weight, float[21] model_classifier_model_4_bias, float[64,3,7,7] onnx__Conv_525, float[64] onnx__Conv_526, float[64,64,1,1] onnx__Conv_528, float[64] onnx__Conv_529, float[64,64,3,3] onnx__Conv_531, float[64] onnx__Conv_532, float[256,64,1,1] onnx__Conv_534, float[256] onnx__Conv_535, float[256,64,1,1] onnx__Conv_537, float[256] onnx__Conv_538, float[64,256,1,1] onnx__Conv_540, float[64] onnx__Conv_541, float[64,64,3,3] onnx__Conv_543, float[64] onnx__Conv_544, float[256,64,1,1] onnx__Conv_546, float[256] onnx__Conv_547, float[64,256,1,1] onnx__Conv_549, float[64] onnx__Conv_550, float[64,64,3,3] onnx__Conv_552, float[64] onnx__Conv_553, float[256,64,1,1] onnx__Conv_555, float[256] onnx__Conv_556, float[128,256,1,1] onnx__Conv_558, float[128] onnx__Conv_559, float[128,128,3,3] onnx__Conv_561, float[128] onnx__Conv_562, float[512,128,1,1] onnx__Conv_564, float[512] onnx__Conv_565, float[512,256,1,1] onnx__Conv_567, float[512] onnx__Conv_568, float[128,512,1,1] onnx__Conv_570, float[128] onnx__Conv_571, float[128,128,3,3] onnx__Conv_573, float[128] onnx__Conv_574, float[512,128,1,1] onnx__Conv_576, float[512] onnx__Conv_577, float[128,512,1,1] onnx__Conv_579, float[128] onnx__Conv_580, float[128,128,3,3] onnx__Conv_582, float[128] onnx__Conv_583, float[512,128,1,1] onnx__Conv_585, float[512] onnx__Conv_586, float[128,512,1,1] onnx__Conv_588, float[128] onnx__Conv_589, float[128,128,3,3] onnx__Conv_591, float[128] onnx__Conv_592, float[512,128,1,1] onnx__Conv_594, float[512] onnx__Conv_595, float[256,512,1,1] onnx__Conv_597, float[256] onnx__Conv_598, float[256,256,3,3] onnx__Conv_600, float[256] onnx__Conv_601, float[1024,256,1,1] onnx__Conv_603, float[1024] onnx__Conv_604, float[1024,512,1,1] onnx__Conv_606, float[1024] onnx__Conv_607, float[256,1024,1,1] onnx__Conv_609, float[256] onnx__Conv_610, float[256,256,3,3] onnx__Conv_612, float[256] onnx__Conv_613, float[1024,256,1,1] onnx__Conv_615, float[1024] onnx__Conv_616, float[256,1024,1,1] onnx__Conv_618, float[256] onnx__Conv_619, float[256,256,3,3] onnx__Conv_621, float[256] onnx__Conv_622, float[1024,256,1,1] onnx__Conv_624, float[1024] onnx__Conv_625, float[256,1024,1,1] onnx__Conv_627, float[256] onnx__Conv_628, float[256,256,3,3] onnx__Conv_630, float[256] onnx__Conv_631, float[1024,256,1,1] onnx__Conv_633, float[1024] onnx__Conv_634, float[256,1024,1,1] onnx__Conv_636, float[256] onnx__Conv_637, float[256,256,3,3] onnx__Conv_639, float[256] onnx__Conv_640, float[1024,256,1,1] onnx__Conv_642, float[1024] onnx__Conv_643, float[256,1024,1,1] onnx__Conv_645, float[256] onnx__Conv_646, float[256,256,3,3] onnx__Conv_648, float[256] onnx__Conv_649, float[1024,256,1,1] onnx__Conv_651, float[1024] onnx__Conv_652, float[512,1024,1,1] onnx__Conv_654, float[512] onnx__Conv_655, float[512,512,3,3] onnx__Conv_657, float[512] onnx__Conv_658, float[2048,512,1,1] onnx__Conv_660, float[2048] onnx__Conv_661, float[2048,1024,1,1] onnx__Conv_663, float[2048] onnx__Conv_664, float[512,2048,1,1] onnx__Conv_666, float[512] onnx__Conv_667, float[512,512,3,3] onnx__Conv_669, float[512] onnx__Conv_670, float[2048,512,1,1] onnx__Conv_672, float[2048] onnx__Conv_673, float[512,2048,1,1] onnx__Conv_675, float[512] onnx__Conv_676, float[512,512,3,3] onnx__Conv_678, float[512] onnx__Conv_679, float[2048,512,1,1] onnx__Conv_681, float[2048] onnx__Conv_682, float[512,2048,3,3] onnx__Conv_684, float[512] onnx__Conv_685) => (float[resize_output_0000_dynamic_axes_1,Resizeresize_output_0000_dim_1,Resizeresize_output_0000_dim_2,Resizeresize_output_0000_dim_3] output_var) {
       shape_output_0000 = Shape (input_1)
       constant_output_0000 = Constant <value = int64 {2}> ()
       gather_output_0000 = Gather <axis = 0> (shape_output_0000, constant_output_0000)
       shape_output_0001 = Shape (input_1)
       constant_output_0001 = Constant <value = int64 {3}> ()
       gather_output_0001 = Gather <axis = 0> (shape_output_0001, constant_output_0001)
       conv_output_0000 = Conv <dilations = [1, 1], group = 1, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]> (input_1, onnx__Conv_525, onnx__Conv_526)
       relu_output_0000 = Relu (conv_output_0000)
       maxpool_output_0000 = MaxPool <ceil_mode = 0, dilations = [1, 1], kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]> (relu_output_0000)
       conv_output_0001 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (maxpool_output_0000, onnx__Conv_528, onnx__Conv_529)
       relu_output_0001 = Relu (conv_output_0001)
       conv_output_0002 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0001, onnx__Conv_531, onnx__Conv_532)
       relu_output_0002 = Relu (conv_output_0002)
       conv_output_0003 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0002, onnx__Conv_534, onnx__Conv_535)
       conv_output_0004 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (maxpool_output_0000, onnx__Conv_537, onnx__Conv_538)
       add_output_0000 = Add (conv_output_0003, conv_output_0004)
       relu_output_0003 = Relu (add_output_0000)
       conv_output_0005 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0003, onnx__Conv_540, onnx__Conv_541)
       relu_output_0004 = Relu (conv_output_0005)
       conv_output_0006 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0004, onnx__Conv_543, onnx__Conv_544)
       relu_output_0005 = Relu (conv_output_0006)
       conv_output_0007 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0005, onnx__Conv_546, onnx__Conv_547)
       add_output_0001 = Add (conv_output_0007, relu_output_0003)
       relu_output_0006 = Relu (add_output_0001)
       conv_output_0008 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0006, onnx__Conv_549, onnx__Conv_550)
       relu_output_0007 = Relu (conv_output_0008)
       conv_output_0009 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0007, onnx__Conv_552, onnx__Conv_553)
       relu_output_0008 = Relu (conv_output_0009)
       conv_output_0010 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0008, onnx__Conv_555, onnx__Conv_556)
       add_output_0002 = Add (conv_output_0010, relu_output_0006)
       relu_output_0009 = Relu (add_output_0002)
       conv_output_0011 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0009, onnx__Conv_558, onnx__Conv_559)
       relu_output_0010 = Relu (conv_output_0011)
       conv_output_0012 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]> (relu_output_0010, onnx__Conv_561, onnx__Conv_562)
       relu_output_0011 = Relu (conv_output_0012)
       conv_output_0013 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0011, onnx__Conv_564, onnx__Conv_565)
       conv_output_0014 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [2, 2]> (relu_output_0009, onnx__Conv_567, onnx__Conv_568)
       add_output_0003 = Add (conv_output_0013, conv_output_0014)
       relu_output_0012 = Relu (add_output_0003)
       conv_output_0015 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0012, onnx__Conv_570, onnx__Conv_571)
       relu_output_0013 = Relu (conv_output_0015)
       conv_output_0016 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0013, onnx__Conv_573, onnx__Conv_574)
       relu_output_0014 = Relu (conv_output_0016)
       conv_output_0017 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0014, onnx__Conv_576, onnx__Conv_577)
       add_output_0004 = Add (conv_output_0017, relu_output_0012)
       relu_output_0015 = Relu (add_output_0004)
       conv_output_0018 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0015, onnx__Conv_579, onnx__Conv_580)
       relu_output_0016 = Relu (conv_output_0018)
       conv_output_0019 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0016, onnx__Conv_582, onnx__Conv_583)
       relu_output_0017 = Relu (conv_output_0019)
       conv_output_0020 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0017, onnx__Conv_585, onnx__Conv_586)
       add_output_0005 = Add (conv_output_0020, relu_output_0015)
       relu_output_0018 = Relu (add_output_0005)
       conv_output_0021 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0018, onnx__Conv_588, onnx__Conv_589)
       relu_output_0019 = Relu (conv_output_0021)
       conv_output_0022 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0019, onnx__Conv_591, onnx__Conv_592)
       relu_output_0020 = Relu (conv_output_0022)
       conv_output_0023 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0020, onnx__Conv_594, onnx__Conv_595)
       add_output_0006 = Add (conv_output_0023, relu_output_0018)
       relu_output_0021 = Relu (add_output_0006)
       conv_output_0024 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0021, onnx__Conv_597, onnx__Conv_598)
       relu_output_0022 = Relu (conv_output_0024)
       conv_output_0025 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0022, onnx__Conv_600, onnx__Conv_601)
       relu_output_0023 = Relu (conv_output_0025)
       conv_output_0026 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0023, onnx__Conv_603, onnx__Conv_604)
       conv_output_0027 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0021, onnx__Conv_606, onnx__Conv_607)
       add_output_0007 = Add (conv_output_0026, conv_output_0027)
       relu_output_0024 = Relu (add_output_0007)
       conv_output_0028 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0024, onnx__Conv_609, onnx__Conv_610)
       relu_output_0025 = Relu (conv_output_0028)
       conv_output_0029 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0025, onnx__Conv_612, onnx__Conv_613)
       relu_output_0026 = Relu (conv_output_0029)
       conv_output_0030 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0026, onnx__Conv_615, onnx__Conv_616)
       add_output_0008 = Add (conv_output_0030, relu_output_0024)
       relu_output_0027 = Relu (add_output_0008)
       conv_output_0031 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0027, onnx__Conv_618, onnx__Conv_619)
       relu_output_0028 = Relu (conv_output_0031)
       conv_output_0032 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0028, onnx__Conv_621, onnx__Conv_622)
       relu_output_0029 = Relu (conv_output_0032)
       conv_output_0033 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0029, onnx__Conv_624, onnx__Conv_625)
       add_output_0009 = Add (conv_output_0033, relu_output_0027)
       relu_output_0030 = Relu (add_output_0009)
       conv_output_0034 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0030, onnx__Conv_627, onnx__Conv_628)
       relu_output_0031 = Relu (conv_output_0034)
       conv_output_0035 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0031, onnx__Conv_630, onnx__Conv_631)
       relu_output_0032 = Relu (conv_output_0035)
       conv_output_0036 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0032, onnx__Conv_633, onnx__Conv_634)
       add_output_0010 = Add (conv_output_0036, relu_output_0030)
       relu_output_0033 = Relu (add_output_0010)
       conv_output_0037 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0033, onnx__Conv_636, onnx__Conv_637)
       relu_output_0034 = Relu (conv_output_0037)
       conv_output_0038 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0034, onnx__Conv_639, onnx__Conv_640)
       relu_output_0035 = Relu (conv_output_0038)
       conv_output_0039 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0035, onnx__Conv_642, onnx__Conv_643)
       add_output_0011 = Add (conv_output_0039, relu_output_0033)
       relu_output_0036 = Relu (add_output_0011)
       conv_output_0040 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0036, onnx__Conv_645, onnx__Conv_646)
       relu_output_0037 = Relu (conv_output_0040)
       conv_output_0041 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0037, onnx__Conv_648, onnx__Conv_649)
       relu_output_0038 = Relu (conv_output_0041)
       conv_output_0042 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0038, onnx__Conv_651, onnx__Conv_652)
       add_output_0012 = Add (conv_output_0042, relu_output_0036)
       relu_output_0039 = Relu (add_output_0012)
       conv_output_0043 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0039, onnx__Conv_654, onnx__Conv_655)
       relu_output_0040 = Relu (conv_output_0043)
       conv_output_0044 = Conv <dilations = [2, 2], group = 1, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]> (relu_output_0040, onnx__Conv_657, onnx__Conv_658)
       relu_output_0041 = Relu (conv_output_0044)
       conv_output_0045 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0041, onnx__Conv_660, onnx__Conv_661)
       conv_output_0046 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0039, onnx__Conv_663, onnx__Conv_664)
       add_output_0013 = Add (conv_output_0045, conv_output_0046)
       relu_output_0042 = Relu (add_output_0013)
       conv_output_0047 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0042, onnx__Conv_666, onnx__Conv_667)
       relu_output_0043 = Relu (conv_output_0047)
       conv_output_0048 = Conv <dilations = [4, 4], group = 1, kernel_shape = [3, 3], pads = [4, 4, 4, 4], strides = [1, 1]> (relu_output_0043, onnx__Conv_669, onnx__Conv_670)
       relu_output_0044 = Relu (conv_output_0048)
       conv_output_0049 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0044, onnx__Conv_672, onnx__Conv_673)
       add_output_0014 = Add (conv_output_0049, relu_output_0042)
       relu_output_0045 = Relu (add_output_0014)
       conv_output_0050 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0045, onnx__Conv_675, onnx__Conv_676)
       relu_output_0046 = Relu (conv_output_0050)
       conv_output_0051 = Conv <dilations = [4, 4], group = 1, kernel_shape = [3, 3], pads = [4, 4, 4, 4], strides = [1, 1]> (relu_output_0046, onnx__Conv_678, onnx__Conv_679)
       relu_output_0047 = Relu (conv_output_0051)
       conv_output_0052 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0047, onnx__Conv_681, onnx__Conv_682)
       add_output_0015 = Add (conv_output_0052, relu_output_0045)
       relu_output_0048 = Relu (add_output_0015)
       conv_output_0053 = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]> (relu_output_0048, onnx__Conv_684, onnx__Conv_685)
       relu_output_0049 = Relu (conv_output_0053)
       conv_output_0054 = Conv <dilations = [1, 1], group = 1, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]> (relu_output_0049, model_classifier_model_4_weight, model_classifier_model_4_bias)
       constant_output_0002 = Constant <value = int64[1] {0}> ()
       unsqueeze_output_0000 = Unsqueeze (gather_output_0000, constant_output_0002)
       constant_output_0003 = Constant <value = int64[1] {0}> ()
       unsqueeze_output_0001 = Unsqueeze (gather_output_0001, constant_output_0003)
       concat_output_0000 = Concat <axis = 0> (unsqueeze_output_0000, unsqueeze_output_0001)
       shape_output_0002 = Shape (conv_output_0054)
       constant_output_0004 = Constant <value = int64[1] {0}> ()
       constant_output_0005 = Constant <value = int64[1] {0}> ()
       constant_output_0006 = Constant <value = int64[1] {2}> ()
       slice_output_0000 = Slice (shape_output_0002, constant_output_0005, constant_output_0006, constant_output_0004)
       cast_output_0000 = Cast <to = 7> (concat_output_0000)
       concat_output_0001 = Concat <axis = 0> (slice_output_0000, cast_output_0000)
       output_var = Resize <coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -0.75, mode = "linear", nearest_mode = "floor"> (conv_output_0054, , , concat_output_0001)
    }
    import torch, onnx2torch
    from torch import tensor
    
    class Model(torch.nn.Module):
        
        def __init__(self):
            super().__init__()
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
            setattr(self,'Conv_53', torch.nn.modules.conv.Conv2d(**{'in_channels':2048,'out_channels':512,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
            setattr(self,'Relu_49', torch.nn.modules.activation.ReLU(**{'inplace':False}))
            setattr(self,'Conv_54', torch.nn.modules.conv.Conv2d(**{'in_channels':512,'out_channels':21,'kernel_size':(1, 1),'stride':(1, 1),'padding':(0, 0),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
            setattr(self,'Constant_2', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
            setattr(self,'Unsqueeze', onnx2torch.node_converters.unsqueeze.OnnxUnsqueezeStaticAxes(**{'axes':[0]}))
            setattr(self,'Constant_3', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
            setattr(self,'Unsqueeze_1', onnx2torch.node_converters.unsqueeze.OnnxUnsqueezeStaticAxes(**{'axes':[0]}))
            setattr(self,'Concat', onnx2torch.node_converters.concat.OnnxConcat(**{'axis':0}))
            setattr(self,'Shape_2', onnx2torch.node_converters.shape.OnnxShape(**{'start':0,'end':None}))
            setattr(self,'Constant_4', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
            setattr(self,'Constant_5', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.zeros((1,))}))
            setattr(self,'Constant_6', onnx2torch.node_converters.constant.OnnxConstant(**{'value':torch.ones((1,))*2}))
            setattr(self,'Slice', onnx2torch.node_converters.slice.OnnxSlice(**{}))
            setattr(self,'Cast', onnx2torch.node_converters.cast.OnnxCast(**{'onnx_dtype':7}))
            setattr(self,'Concat_1', onnx2torch.node_converters.concat.OnnxConcat(**{'axis':0}))
            setattr(self,'Resize', onnx2torch.node_converters.resize.OnnxResize(**{'mode':'linear','align_corners':False,'ignore_roi':True,'ignore_bs_ch_size':False}))
    
        def forward(self, input_1):
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
            conv_53 = self.Conv_53(relu_48);  relu_48 = None
            relu_49 = self.Relu_49(conv_53);  conv_53 = None
            conv_54 = self.Conv_54(relu_49);  relu_49 = None
            constant_2 = self.Constant_2()
            unsqueeze = self.Unsqueeze(gather);  gather = None
            constant_3 = self.Constant_3()
            unsqueeze_1 = self.Unsqueeze_1(gather_1);  gather_1 = None
            concat = self.Concat(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
            shape_2 = self.Shape_2(conv_54)
            constant_4 = self.Constant_4()
            constant_5 = self.Constant_5()
            constant_6 = self.Constant_6()
            slice_1 = self.Slice(shape_2, constant_5, constant_6, constant_4);  shape_2 = constant_5 = constant_6 = constant_4 = None
            cast = self.Cast(concat);  concat = None
            concat_1 = self.Concat_1(slice_1, cast);  slice_1 = cast = None
            resize = self.Resize(conv_54, sizes = concat_1);  conv_54 = concat_1 = None
            return resize
        
    cpu
    cpu cpu




.. parsed-literal::

    torch.Size([16, 21, 300, 300])



3. While you train the model, Modlee prepares everything you need for your convenience:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Assumes that modlee_model is 
    import inspect
    
    class RecommendedModel(modlee.recommender.RecommendedModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def configure_optimizers(self,):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=0.001,
            )
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.8,
                patience=200,
            )
            return optimizer
        
        def on_train_epoch_end(self) -> None:
            """
            Update the learning rate scheduler
            """
            sch = self.scheduler
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(self.trainer.callback_metrics["loss"])
                self.log('scheduler_last_lr',sch._last_lr[0])
            return super().on_train_epoch_end()
        
    recd_model = RecommendedModel(modlee_model)
    
    # The built-in configure callbacks function should be the same as the base ModleeModel
    print("==== ORIGINAL configure_callbacks ====")
    print(inspect.getsource(recd_model.configure_callbacks))
    # The updated configure_optimizers, with patience of 200, should be printed
    print("==== ORIGINAL configure_optimizers ====")
    print(inspect.getsource(modlee.recommender.RecommendedModel.configure_optimizers))
    print("==== UPDATED configure_optimizers ====")
    print(inspect.getsource(recd_model.configure_optimizers))


.. parsed-literal::

    ==== ORIGINAL configure_callbacks ====
        def configure_callbacks(self):
            base_callbacks = super().configure_callbacks()
            # base_callbacks.append(
            #     pl.callbacks.EarlyStopping(
            #         'val_loss',
            #         patience=10,
            #         verbose=True,)
            # )
            return base_callbacks
    
    ==== ORIGINAL configure_optimizers ====
        def configure_optimizers(self,):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=0.001,
            )
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.8,
                patience=10,
            )
            return optimizer
    
    ==== ORIGINAL configure_optimizers ====
        def configure_optimizers(self,):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=0.001,
            )
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.8,
                patience=200,
            )
            return optimizer
    


.. code:: ipython3

    # print(dir(modlee_model))
    # import inspect
    # print(inspect.getsource(modlee_model.train))
    # import lightning.pytorch as pl
    # callbacks = modlee_model.configure_callbacks()
    # print(callbacks)
    # trainer = pl.Trainer(
    #     max_epochs=1,
    #     # callbacks 2,3,4 (logOutput, logParams, PushAPI) are fine
    #     # callback 0 (dataStats) is fine
    #     # 1 also seems fine
    #     # callbacks=[callbacks[c] for c in [1]], 
    #     callbacks=callbacks,
    #     enable_model_summary=False,
    #     )
    # with modlee.start_run() as run:
    #     trainer.fit(model=modlee_model, 
    #         train_dataloaders=train_dataloader,
    #         val_dataloaders=train_dataloader)
    recommender.train(max_epochs=1, val_dataloaders=train_dataloader)


.. parsed-literal::

    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]


.. parsed-literal::

    ----------------------------------------------------------------
    Training your recommended modlee model:
         - Running this model: ./modlee_model.py
         - On the dataloader previously analyzed by the recommender
    ----------------------------------------------------------------



.. parsed-literal::

    Sanity Checking: 0it [00:00, ?it/s]


.. parsed-literal::

    /home/ubuntu/projects/.venv/lib/python3.10/site-packages/lightning/pytorch/loops/fit_loop.py:281: PossibleUserWarning: The number of training batches (14) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
      rank_zero_warn(



.. parsed-literal::

    Training: 0it [00:00, ?it/s]


.. parsed-literal::

    /home/ubuntu/projects/.venv/lib/python3.10/site-packages/onnx2torch/node_converters/shape.py:46: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      return torch.tensor(
    /home/ubuntu/projects/.venv/lib/python3.10/site-packages/onnx2torch/node_converters/slice.py:33: TracerWarning: Converting a tensor to a NumPy array might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      axes = axes.detach().cpu().numpy()
    /home/ubuntu/projects/.venv/lib/python3.10/site-packages/onnx2torch/node_converters/slice.py:36: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
      steps = [1] * len(starts)
    /home/ubuntu/projects/.venv/lib/python3.10/site-packages/onnx2torch/node_converters/slice.py:42: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
      for start, end, axis, step in zip(starts, ends, axes, steps):
    /home/ubuntu/projects/.venv/lib/python3.10/site-packages/onnx2torch/node_converters/resize.py:73: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if sizes.nelement() != 0:
    /home/ubuntu/projects/.venv/lib/python3.10/site-packages/onnx2torch/node_converters/resize.py:74: TracerWarning: Converting a tensor to a Python list might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      sizes = sizes.tolist()
    /home/ubuntu/projects/.venv/lib/python3.10/site-packages/onnx2torch/node_converters/resize.py:76: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if not self.ignore_bs_ch_size and input_shape[:2] != sizes[:2]:



.. parsed-literal::

    Validation: 0it [00:00, ?it/s]


.. parsed-literal::

    INFO: Metric val_loss improved. New best score: 16.994
    INFO:lightning.pytorch.callbacks.early_stopping:Metric val_loss improved. New best score: 16.994


4. Modlee auto-documents your experiment locally and learns from non-sensitive details:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sharing helps to enhance ML model recommendations across the entire
community of modlee users

.. code:: ipython3

    recommender.train_documentation_locations()


.. parsed-literal::

    
    -----------------------------------------------------------------------------------------------
    
    Modlee documented all the details about your trained model and experiment here: 
    
            Path: /home/ubuntu/projects/modlee_survey/notebooks/mlruns/0/23cd9c1a052c49a88e1b73c22a4ad574/
            Experiment_id: automatically assigned to | 0
            Run_id: automatically assigned to | 23cd9c1a052c49a88e1b73c22a4ad574
    
    -----------------------------------------------------------------------------------------------
    


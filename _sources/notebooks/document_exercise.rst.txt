Documentation
=============

In this exercise, you will implement the ``modlee`` package to document
an image segmentation experiment.

.. code:: ipython3

    # Boilerplate imports
    import lightning.pytorch as pl
    import torch.nn.functional as F
    import torch.nn as nn
    import torch
    import torchvision
    from torchvision.transforms.functional import InterpolationMode
    import os
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

In the next cell, import ``modlee`` and initialize with an API key.

.. code:: ipython3

    # Your code goes here. Import the modlee package and initialize with your API key.


Load the training data.

.. code:: ipython3

    train_loader, val_loader = get_fashion_mnist()
    num_classes = len(train_loader.dataset.classes)
    
    train_dataset, val_dataset = torchvision.datasets.VOCSegmentation(
        root='./', download=True,)
    
    
    def replace_tensor_value_(tensor, a, b):
        tensor[tensor == a] = b
        return tensor
    
    input_resize = transforms.Resize((224, 224))
    input_transform = transforms.Compose(
        [
            input_resize,
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    
    target_resize = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
    target_transform = transforms.Compose(
        [
            target_resize,
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: replace_tensor_value_(x.squeeze(0).long(), 255, 21)),
        ]
    )
    
    # Creating the dataset
    train_dataset = torchvision.datasets.VOCSegmentation(
        './datasets/',
        year='2007',
        download=True,
        image_set='train',
        transform=input_transform,
        target_transform=target_transform,
    )
    val_dataset = torchvision.datasets.VOCSegmentation(
        './datasets/',
        year='2007',
        download=True,
        image_set='val',
        transform=input_transform,
        target_transform=target_transform,
    )
    
    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


.. parsed-literal::

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz


.. parsed-literal::

    100%|██████████| 26421880/26421880 [00:01<00:00, 13333207.84it/s]


.. parsed-literal::

    Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz


.. parsed-literal::

    100%|██████████| 29515/29515 [00:00<00:00, 332473.60it/s]


.. parsed-literal::

    Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz


.. parsed-literal::

    100%|██████████| 4422102/4422102 [00:00<00:00, 6112706.99it/s]


.. parsed-literal::

    Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz


.. parsed-literal::

    100%|██████████| 5148/5148 [00:00<00:00, 6405303.17it/s]

.. parsed-literal::

    Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
    


.. parsed-literal::

    


Build the PyTorch model as a ``torch.nn.Module``.

.. code:: ipython3

    class Classifier(torch.nn.Module):
        def __init__(self, num_classes=10):
            super(Classifier, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            x = F.softmax(x)
            return x

.. code:: ipython3

    class conv1x1_block(nn.Module):
        '''(conv 1 x 1 )'''
        def __init__(self, in_planes, out_planes, stride = 1):
            super(conv1x1_block, self).__init__()
            self.conv =  nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding='same')
            
    
        def forward(self, x):
            x = self.conv(x)
            return x 
    
    class conv3x3_block_x1(nn.Module):
        '''(conv => BN => ReLU) * 1'''
    
        def __init__(self, in_ch, out_ch):
            super(conv3x3_block_x1, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding='same'),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
    
        def forward(self, x):
            x = self.conv(x)
            return x
    
    class conv3x3_block_x2(nn.Module):
        '''(conv => BN => ReLU) * 2'''
    
        def __init__(self, in_ch, out_ch):
            super(conv3x3_block_x2, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding='same'),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding='same'),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
    
        def forward(self, x):
            x = self.conv(x)
            return x
    
    class upsample(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(upsample, self).__init__()
            self.conv1x1 = conv1x1_block(in_ch, out_ch)
            self.conv = conv3x3_block_x2(in_ch, out_ch)
    
        def forward(self, H, L):
            """
            H: High level feature map, upsample
            L: Low level feature map, block output
            """
            H = F.interpolate(H, scale_factor=2, mode='bilinear', align_corners=False)
            H = self.conv1x1(H)
            x = torch.cat([H, L], dim=1)
            x = self.conv(x)
            return x
    
    class UNet(nn.Module):
        def __init__(self, num_classes=22):
            super(UNet, self).__init__()
            self.maxpool = nn.MaxPool2d(2)
            self.block1 = conv3x3_block_x2(3, 64)
            self.block2 = conv3x3_block_x2(64, 128)
            self.block3 = conv3x3_block_x2(128, 256)
            self.block4 = conv3x3_block_x2(256, 512)
            self.block_out = conv3x3_block_x1(256, 512)
            self.upsample1 = upsample(1024, 512)
            self.upsample2 = upsample(512, 256)
            self.upsample3 = upsample(256, 128)
            self.upsample4 = upsample(128, 64)
            self.upsample_out = conv3x3_block_x2(64, num_classes)
    
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
    
        def forward(self, x):
            block1_x = self.block1(x)
            x = self.maxpool(block1_x)
            block2_x = self.block2(x)
            x = self.maxpool(block2_x)
            block3_x = self.block3(x)
            x = self.maxpool(block3_x)
            x = self.block_out(x)
            x = self.upsample2(x, block3_x)
            x = self.upsample3(x, block2_x)
            x = self.upsample4(x, block1_x)
            x = self.upsample_out(x)
    
            return x

In the next cell, wrap the model defined above in a
``modlee.model.ModleeModel`` object. At minimum, you must define the
``__init__()``, ``forward()``, ``training_step()``, and
``configure_optimizers()`` functions. Refer to the `Lightning
documentation <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>`__
for a refresher.

.. code:: ipython3

    class ModleeUNet( '''Inherit the modlee model''' )
        def __init__(self):                # Fill out the constructor
            # Fill out the constructor
            pass
        
        def forward(self, x):
            # Fill out the forward pass
            pass
        
        def training_step(self, batch, batch_idx):
            # Fill out the training step
            pass
        
        def configure_optimizers(self):
            # Fill out the optimizer configuration
            pass
        
    model = ModleeUNet()

In the next cell, start training within a ``modlee.start_run()``
`context manager <https://realpython.com/python-with-statement/>`__.
Refer to ```mlflow``\ ’s
implementation <https://mlflow.org/docs/latest/python_api/mlflow.html>`__
as a refresher.

.. code:: ipython3

    # Your code goes here. Star training within a modlee.start_run() context manager


.. parsed-literal::

    Missing logger folder: /home/ubuntu/projects/modlee_pypi/examples/mlruns/0/297bc8969bb64235bbfa0824f20dd24b/artifacts/mlruns/0/0729922d48c14c8b9c78f3b15ca962e3/artifacts/lightning_logs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name       | Type       | Params
    ------------------------------------------
    0 | classifier | Classifier | 44.4 K
    ------------------------------------------
    44.4 K    Trainable params
    0         Non-trainable params
    44.4 K    Total params
    0.178     Total estimated model params size (MB)



.. parsed-literal::

    Sanity Checking: 0it [00:00, ?it/s]



.. parsed-literal::

    Training: 0it [00:00, ?it/s]



.. parsed-literal::

    Validation: 0it [00:00, ?it/s]


``modlee`` with ``mlflow`` underneath will document the experiment in an
automatically generated ``assets`` folder.

.. code:: ipython3

    last_run_path = modlee.last_run_path()
    print(f"Run path: {last_run_path}")
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    artifacts = os.listdir(artifacts_path)
    print(f"Saved artifacts: {artifacts}")


.. parsed-literal::

    Run path: /home/ubuntu/projects/modlee_pypi/examples/mlruns/0/297bc8969bb64235bbfa0824f20dd24b/artifacts/mlruns/0/0729922d48c14c8b9c78f3b15ca962e3/artifacts/mlruns/0/910e72fc9bef4e958046ffc5fe3e3585
    Saved artifacts: ['model_graph.py', 'model_graph.txt', 'model_size', 'model', 'cached_vars', 'stats_rep', 'snapshot_1.npy', 'snapshot_0.npy', 'model.py', 'loss_calls.txt', 'model_summary.txt']


We can build the model from the cached ``model_graph.Model`` class and
confirm that we can pass an input through it. Note that this model’s
weights will be uninitialized. To load the model from the last
checkpoint, we can load it directly from the cached ``model.pth``.

Rebuild the saved model. First, determine the path to the most recent
run.

.. code:: ipython3

    last_run_path = # Get the most recent run
    artifacts_path = os.path.join(last_run_path, 'artifacts')

Next, import the model from the assets saved in the ``artifacts/``
directory.

.. code:: ipython3

    os.chdir(artifacts_path)
    
    import # the model graph
    rebuilt_model = # Construct the model
    
    # Pass an input through the model
    x, _ = next(iter(train_loader))
    y_rebuilt = rebuilt_model(x)

You’ve reached the end of the tutorial and can now implement ``modlee``
into your machine learning experiments. Congratulations!

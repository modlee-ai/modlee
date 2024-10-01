Documentation
=============

In this exercise, you will implement the ``modlee`` package to document
an image segmentation experiment with a pretrained model from
``torchvision``.

.. code:: ipython3

    # Boilerplate imports
    import lightning.pytorch as pl
    import torch.nn.functional as F
    import torch.nn as nn
    import torch
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    import os
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

In the next cell, import ``modlee`` and initialize with an API key.

.. code:: ipython3

    # Your code goes here. Import the modlee package and initialize with your API key.
    os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
    import modlee
    modlee.init(api_key="modleemichael")

Load the training data.

.. code:: ipython3

    imagenet_mean = [0.485, 0.456, 0.406]  # mean of the imagenet dataset for normalizing
    imagenet_std = [0.229, 0.224, 0.225]  # std of the imagenet dataset for normalizing
    
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
        image_set='val',
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

    Using downloaded and verified file: ./datasets/VOCtrainval_06-Nov-2007.tar
    Extracting ./datasets/VOCtrainval_06-Nov-2007.tar to ./datasets/
    Using downloaded and verified file: ./datasets/VOCtrainval_06-Nov-2007.tar
    Extracting ./datasets/VOCtrainval_06-Nov-2007.tar to ./datasets/


Create the image segmentation model using a `pretrained fully connected
network <https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html#torchvision.models.segmentation.fcn_resnet50>`__.

.. code:: ipython3

    model = torchvision.models.segmentation.fcn_resnet50(num_classes=22)

In the next cell, wrap the model defined above in a
``modlee.model.ModleeModel`` object. At minimum, you must define the
``__init__()``, ``forward()``, ``training_step()``, and
``configure_optimizers()`` functions. Refer to the `Lightning
documentation <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>`__
for a refresher.

.. code:: ipython3

    class ModleeFCN(modlee.model.ModleeModel):
        def __init__(self):                # Fill out the constructor
            # Fill out the constructor
            super().__init__()
            self.model = model
            pass
        
        def forward(self, x):
            # Fill out the forward pass
            return self.model(x)
            pass
        
        def training_step(self, batch, batch_idx):
            # Fill out the training step
            x, y_target = batch
            
            y_pred = self(x)['out']
            # print(y_pred)
            loss = F.cross_entropy(y_pred, y_target)
            return loss
            pass
        
        def configure_optimizers(self):
            # Fill out the optimizer configuration
            return torch.optim.Adam(
                self.parameters(), 
                lr=0.001,
            )
            pass
        
    model = ModleeFCN()

In the next cell, start training within a ``modlee.start_run()``
`context manager <https://realpython.com/python-with-statement/>`__.
Refer to ```mlflow``\ ’s
implementation <https://mlflow.org/docs/latest/python_api/mlflow.html>`__
as a refresher.

.. code:: ipython3

    # Your code goes here. Star training within a modlee.start_run() context manager
    with modlee.start_run() as run:
        trainer = modlee.Trainer(max_epochs=1)
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
        )


.. parsed-literal::

    Missing logger folder: /home/ubuntu/projects/modlee_pypi/examples/mlruns/0/7a47086681324d0e924f9076a1262de9/artifacts/lightning_logs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name  | Type | Params
    -------------------------------
    0 | model | FCN  | 33.0 M
    -------------------------------
    33.0 M    Trainable params
    0         Non-trainable params
    33.0 M    Total params
    131.830   Total estimated model params size (MB)
    /opt/conda/envs/modlee/lib/python3.10/site-packages/lightning/pytorch/loops/fit_loop.py:281: PossibleUserWarning: The number of training batches (14) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
      rank_zero_warn(



.. parsed-literal::

    Training: 0it [00:00, ?it/s]


.. parsed-literal::

    WARNING:root:Cannot log output shape, could not pass batch through network


Rebuild the saved model. First, determine the path to the most recent
run.

.. code:: ipython3

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    print(os.listdir(artifacts_path))
    print(os.path.join(artifacts_path,'model_graph.py'))


.. parsed-literal::

    ['transforms.txt', 'model_graph.py', 'model_graph.txt', 'model_size', 'model', 'cached_vars', 'stats_rep', 'snapshot_1.npy', 'lightning_logs', 'snapshot_0.npy', 'model.py', 'loss_calls.txt', 'model_summary.txt']
    /home/ubuntu/projects/modlee_pypi/examples/mlruns/0/7a47086681324d0e924f9076a1262de9/artifacts/model_graph.py


Next, import the model from the assets saved in the ``artifacts/``
directory.

.. code:: ipython3

    exercise_dir = os.path.abspath(os.getcwd())
    os.chdir(artifacts_path)
    
    import model_graph
    rebuilt_model = model_graph.Model()
    rebuilt_model.eval()
    
    os.chdir(exercise_dir)
    # Pass an input through the model
    x, _ = next(iter(train_loader))
    with torch.no_grad():
        y_rebuilt = rebuilt_model(x)

You’ve reached the end of the tutorial and can now implement ``modlee``
into your machine learning experiments. Congratulations!

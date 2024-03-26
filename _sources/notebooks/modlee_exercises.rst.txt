**Introduction**
================

In this exercise, you will implement the ``modlee`` package to: -
Document an image segmentation experiment with a pretrained model from
``torchvision``. - Receive and train a recommended model.

It may be helpful to `keep the API documentation
open <https://www.documentation.modlee.ai/index.html>`__.

For best performance, ensure that the runtime is set to use a GPU
(``Runtime > Change runtime type > T4 GPU``).

**Installation**
================

First, we need to install ``modlee`` and its related packages. Make sure
that you have an account and API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__. This process may take a
few minutes, so you can `review the
examples <https://www.documentation.modlee.ai/notebooks/document.html>`__
while waiting.

.. code:: ipython3

    import os
    os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
    !curl -H X-API-KEY:$MODLEE_API_KEY https://server.modlee.ai:7070/get_wheel/requirements.txt -O
    !pip3 install -r requirements.txt

.. code:: ipython3

    !curl -H X-API-KEY:$MODLEE_API_KEY https://server.modlee.ai:7070/get_wheel/modlee-0.0.1.post6-py3-none-any.whl -O
    !curl -H X-API-KEY:$MODLEE_API_KEY https://server.modlee.ai:7070/get_wheel/onnx2torch-1.5.11-py3-none-any.whl -O
    !curl -H X-API-KEY:$MODLEE_API_KEY https://server.modlee.ai:7070/get_wheel/onnx_graphsurgeon-0.3.27-py2.py3-none-any.whl -O
    !pip3 install --force-reinstall --no-deps modlee-0.0.1.post6-py3-none-any.whl onnx2torch-1.5.11-py3-none-any.whl onnx_graphsurgeon-0.3.27-py2.py3-none-any.whl \
        lightning==2.0.7 pytorch-lightning==2.0.7 lightning-utilities lightning-cloud torchmetrics==1.3.2

.. code:: ipython3

    # Boilerplate imports
    import os
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    import lightning.pytorch as pl

**Documentation**
=================

We’re ready to implement modlee to document an image segmentation
experiment. Please `review the documentation
example <https://www.documentation.modlee.ai/notebooks/document.html>`__
before continuing.

In the next cell, import ``modlee`` and initialize with an API key.

.. code:: ipython3

    # Your code goes here. Import the modlee package and initialize with your API key.
    # Import
    import
    # Initialize


Load the training data. This cell requires no modifications.

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

In the next cell, we will construct the model. We initialize the model
from a `pretrained fully connected
network <https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html#torchvision.models.segmentation.fcn_resnet50>`__.
We subclass the ``modlee.model.ModleeModel`` parent class so that the
experiment will automatically document. At minimum, you must define the
``__init__()``, ``forward()``, ``training_step()``, and
``configure_optimizers()`` functions.

.. code:: ipython3

    # Use a prerained torchvision Fully Connected Network
    fcn_model = torchvision.models.segmentation.fcn_resnet50(num_classes=22)
    
    # Subclass the correct modlee class
    class ModleeFCN( ''' Replace this with the correct modlee parent class '''):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.model = # Set the above fcn object to self.model
            self.loss_fn = F.cross_entropy
            pass
    
        def forward(self, x):
            # Fill out the forward pass
            # Should return a tensor after it has passed through the model
            pass
    
        def training_step(self, batch, batch_idx):
            # Fill out the training step
            x, y_target = # Get the input and output from the batch
            y_pred = self(x)['out']
            loss = # Calculate the loss between the prediction and target
            return loss
    
        def configure_optimizers(self):
            # Fill out the optimizer configuration
            pass
    
    # Create the model object
    modlee_model = ModleeFCN()

In the next cell, start training within a ``modlee.start_run()`` context
manager.

.. code:: ipython3

    # Your code goes here. Start training within a modlee.start_run() context manager
    # Create the context manager (with ... as ... :)
    with # Fill in the context manager (with ... as ...)
        # Create the trainer object
        trainer =
        # Fit the trainer to the model and the training dataloader
        trainer.fit(
            # Fill in the arguments for training
        )

Rebuild the saved model. First, determine the path to the most recent
run.

.. code:: ipython3

    last_run_path = # Get the last run path
    artifacts_path = os.path.join(last_run_path, 'artifacts')

Next, reload the model from the assets saved in the ``artifacts/``
directory.

.. code:: ipython3

    # Change directories
    exercise_dir = os.path.abspath(os.getcwd())
    os.chdir(artifacts_path)
    
    # Import the model graph:
    
    rebuilt_model = # Construct the model from the model graph module
    # Set the model to evaluation mode to turn off gradients
    rebuilt_model.eval()
    
    os.chdir(exercise_dir)
    # Pass an input through the model
    x, _ = next(iter(train_loader))
    with torch.no_grad():
        y_rebuilt = rebuilt_model(x)
    
    print(f"Rebuilt output shape: {y_rebuilt.shape}")

You have completed the documentation example.

**Recommendation**
==================

We’re ready to implement a modlee-recommended model in an experiment.
Please `review the recommendation
example <https://www.documentation.modlee.ai/notebooks/recommend.html>`__
before continuing.

We can skip the ``modlee`` initialization steps assuming we did so in
the documentation example. First, we create a dataloader from CIFAR10.
This cell does not need any modifications.

.. code:: ipython3

    transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transforms)
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transforms)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16
    )

Create a ``modlee`` recommender object for an image classification task
and fit to the dataset. This process will calculate the dataset
metafeatures to send to the server. The server will return a recommended
model for the dataset assigned to ``recommender.model``.

.. code:: ipython3

    recommender = # Create a recommender object for Image Classification
    recommender.fit(train_dataloader)
    modlee_model = # Extract the model from the recommender

We can train the model as we would a basic ``ModleeModel``, with
automatic documentation of metafeatures.

.. code:: ipython3

    # Create and train the trainer
    with # ... as ...:
        trainer = # Create the trainer
        trainer.fit(
            # Fill in the arguments
        )

Finally, we can view the saved assets from training.

.. code:: ipython3

    last_run_path = # Get the last run path
    print(f"Run path: {last_run_path}")
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    artifacts = os.listdir(artifacts_path)
    print(f"Saved artifacts: {artifacts}")

**Conclusion**
==============

You’ve reached the end of the tutorial and can now implement ``modlee``
into your machine learning experiments. Congratulations!

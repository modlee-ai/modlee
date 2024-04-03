Automate experiment documentation
=================================

This example notebook uses the ``modlee`` package to document a machine
learning experiment with a user-built model. We train a simple
convolutional classifier on the simple Fashion MNIST dataset. After
training, we can reuse the model from the auto-documented model class.
Prerequisites for this tutorial include familiarity with
`PyTorch <https://pytorch.org/docs/stable/index.html>`__ and
`Lightning <https://lightning.ai/docs/pytorch/stable/>`__.

.. code:: ipython3

    # Boilerplate imports
    import os, sys
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import lightning.pytorch as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

Import ``modlee`` and initialize with an API key.

.. code:: ipython3

    # Set the API key to an environment variable,
    # to simulate setting this in your shell profile
    os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
    # Modlee-specific imports
    import modlee
    modlee.init(api_key=os.environ['MODLEE_API_KEY'])

Load the training data; we’ll use ``torch``\ ’s Fashion MNIST dataset.

.. code:: ipython3

    # Get Fashion MNIST, and convert from grayscale to RGB for compatibility with the model
    train_dataloader, val_dataloader = modlee.utils.get_fashion_mnist(num_output_channels=3)
    num_classes = len(train_dataloader.dataset.classes)

Next, we build the model from a pretrained torchvision ResNet model. To
enable automatic documentation, wrap the model in the
``modlee.model.ModleeModel`` class. ``ModleeModel`` subclassees
```lightning.pytorch.LightningModule`` <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`__
and uses the same structure for the ``training_step``,
``validation_step``, and ``configure_optimizers`` functions. Under the
hood, ``ModleeModel`` also contains the callbacks to document the
experiment metafeatures.

.. code:: ipython3

    # Use a pretrained torchvision ResNet
    classifier_model = torchvision.models.resnet18(num_classes=10)
    
    # Subclass the ModleeModel class to enable automatic documentation
    class ModleeClassifier(modlee.model.ModleeModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = classifier_model
            self.loss_fn = F.cross_entropy
    
        def forward(self, x):
            return self.model(x)
    
        def training_step(self, batch, batch_idx):
            x, y_target = batch
            y_pred = self(x)
            loss = self.loss_fn(y_pred, y_target)
            return {"loss": loss}
    
        def validation_step(self, val_batch, batch_idx):
            x, y_target = val_batch
            y_pred = self(x)
            val_loss = self.loss_fn(y_pred, y_target)
            return {'val_loss': val_loss}
            
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
            return optimizer
    
    # Create the model object
    modlee_model = ModleeClassifier()

Run the training loop, just for one epoch.

.. code:: ipython3

    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

::

     | Name  | Type       | Params
   -------------------------------------
   0 | model | Classifier | 44.4 K
   -------------------------------------
   44.4 K    Trainable params
   0         Non-trainable params
   44.4 K    Total params
   0.178     Total estimated model params size (MB)
   Epoch 0: 100%|██████████| 938/938 [00:16<00:00, 57.47it/s, v_num=0]  

``modlee`` with ``mlflow`` underneath will document the experiment in an
automatically generated ``assets`` folder.

.. code:: ipython3

    last_run_path = modlee.last_run_path()
    print(f"Run path: {last_run_path}")
    
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    artifacts = os.listdir(artifacts_path)
    print(f"Saved artifacts: {artifacts}")
    
    os.environ['ARTIFACTS_PATH'] = artifacts_path
    # Add the artifacts directory to the path, 
    # so we can import the model
    sys.path.insert(0, artifacts_path)


::

   Run path: /home/ubuntu/projects/modlee_pypi/examples/mlruns/0/7a47086681324d0e924f9076a1262de9/artifacts/model_graph.py
   Saved artifacts: ['transforms.txt', 'model_graph.py', 'model_graph.txt', 'model_size', 'model', 'cached_vars', 'stats_rep', 'snapshot_1.npy', 'lightning_logs', 'snapshot_0.npy', 'model.py', 'loss_calls.txt', 'model_summary.txt']

.. code:: ipython3

    # Print out the first few lines of the model 
    print("Model graph:")
    !sed -n -e 1,15p $ARTIFACTS_PATH/model_graph.py
    !echo "        ..."
    !sed -n -e 58,68p $ARTIFACTS_PATH/model_graph.py
    !echo "        ..."

::

   Model graph:

   import torch, onnx2torch
   from torch import tensor

   class Model(torch.nn.Module):
       
       def __init__(self):
           super().__init__()
           setattr(self,'Conv', torch.nn.modules.conv.Conv2d(**{'in_channels':3,'out_channels':64,'kernel_size':(7, 7),'stride':(2, 2),'padding':(3, 3),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
           setattr(self,'Relu', torch.nn.modules.activation.ReLU(**{'inplace':False}))
           setattr(self,'MaxPool', torch.nn.modules.pooling.MaxPool2d(**{'kernel_size':[3, 3],'stride':[2, 2],'padding':[1, 1],'dilation':[1, 1],'return_indices':False,'ceil_mode':False}))
           setattr(self,'Conv_1', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':64,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
           setattr(self,'Relu_1', torch.nn.modules.activation.ReLU(**{'inplace':False}))
           setattr(self,'Conv_2', torch.nn.modules.conv.Conv2d(**{'in_channels':64,'out_channels':64,'kernel_size':(3, 3),'stride':(1, 1),'padding':(1, 1),'dilation':(1, 1),'groups':1,'padding_mode':'zeros'}))
           setattr(self,'Add', onnx2torch.node_converters.binary_math_operations.OnnxBinaryMathOperation(**{'operation_type':'Add','broadcast':None,'axis':None}))
           ...

       def forward(self, input_1):
           conv = self.Conv(input_1);  input_1 = None
           relu = self.Relu(conv);  conv = None
           max_pool = self.MaxPool(relu);  relu = None
           conv_1 = self.Conv_1(max_pool)
           relu_1 = self.Relu_1(conv_1);  conv_1 = None
           conv_2 = self.Conv_2(relu_1);  relu_1 = None
           add = self.Add(conv_2, max_pool);  conv_2 = max_pool = None
           relu_2 = self.Relu_2(add);  add = None
           conv_3 = self.Conv_3(relu_2)
           ...

.. code:: ipython3

    # Print the first lines of the data metafeatures
    print("Data metafeatures:")
    !head -20 $ARTIFACTS_PATH/stats_rep

::

   Data metafeatures:
   {
     "dataset_size": 60032,
     "num_sample": 1000,
     "batch_element_0": {
       "raw": {
         "feature_shape": [
           960,
           3,
           28,
           28
         ],
         "stats": {
           "kmeans": {
             "2": {
               "inertia": "155588.50824155417",
               "silhouette_score": "0.19201575",
               "calinski_harabasz_score": "248.3331975601121",
               "davies_bouldin_score": "1.9090644142081366",
               "time_taken": "0.6537415981292725"
             },

We can build the model from the cached ``model_graph.Model`` class and
confirm that we can pass an input through it. Note that this model’s
weights will be uninitialized.

.. code:: ipython3

    # Rebuilding from the object
    import model_graph
    rebuilt_model = model_graph.Model()
    
    # Set models to inference
    modlee_model.eval(); rebuilt_model.eval()


Next, pass an input from the train dataloader through the rebuilt
network and check that the output shape is equal to the original data.

.. code:: ipython3

    
    # Get a batch from the training loader
    x, y = next(iter(train_dataloader))
    with torch.no_grad():
        y_original = modlee_model(x)
        y_rebuilt = rebuilt_model(x)
    assert y_original.shape == y_rebuilt.shape
    
    print(f"Original input and output shapes: {x.shape}, {y_original.shape}")
    print(f"Output shape from module-rebuilt model: {y_rebuilt.shape}")

Alternatively, to load the model from the last checkpoint, we can load
it directly from the cached ``model.pth``.

.. code:: ipython3

    # Reloading from the checkpoint
    reloaded_model = torch.load(os.path.join(artifacts_path, 'model', 'data','model.pth'))
    y_reloaded = reloaded_model(x)
    assert y_original.shape == y_reloaded.shape
    print(f"Output shape from checkpoint-reloaded model: {y_reloaded.shape}")

::

   Original input and output shapes: torch.Size([64, 3, 28, 28]), torch.Size([64, 10])
   Output shape from module-rebuilt model: torch.Size([64, 10])
   Output shape from checkpoint-reloaded model: torch.Size([64, 10])

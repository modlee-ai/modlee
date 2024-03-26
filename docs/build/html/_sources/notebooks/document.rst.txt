Automate experiment documentation
=============

This example notebook uses the ``modlee`` package to document a machine
learning experiment with a user-built model. We train a simple
convolutional classifier on the simple Fashion MNIST dataset. After
training, we can reuse the model from the auto-documented model class.
Prerequisites for this tutorial include familiarity with
`PyTorch <https://pytorch.org/docs/stable/index.html>`__ and
`Lightning <https://lightning.ai/docs/pytorch/stable/>`__.

.. code:: ipython3

    # Boilerplate imports
    import lightning.pytorch as pl
    import torch.nn.functional as F
    import torch.nn as nn
    import torch
    import os
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

Import ``modlee`` and initialize with an API key.

.. code:: ipython3

    # Set the API key to an environment variable,
    # to simulate setting this in your shell profile
    os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
    # Modlee-specific imports
    import modlee
    modlee.init(api_key=os.environ['MODLEE_API_KEY'])

Load the training data.

.. code:: ipython3

    train_loader, val_loader = modlee.utils.get_fashion_mnist()
    num_classes = len(train_loader.dataset.classes)

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
    
    classifier_model = Classifier(num_classes=10)

To enable automatic documentation, wrap the model in the
``modlee.model.ModleeModel`` class. ``ModleeModel`` subclassees
```lightning.pytorch.LightningModule`` <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`__
and uses the same structure for the ``training_step``,
``validation_step``, and ``configure_optimizers`` functions. Under the
hood, ``ModleeModel`` also contains the callbacks to document the
experiment metafeatures.

.. code:: ipython3

    class ModleeClassifier(modlee.model.ModleeModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = classifier_model
            self.loss_fn = F.cross_entropy
    
        def forward(self, x):
            return self.model(x)
    
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_out = self(x)
            loss = F.cross_entropy(y_out, y)
            return {"loss": loss}
    
        def validation_step(self, val_batch, batch_idx):
            x, y_target = val_batch
            y_pred = self(x)
            loss = self.loss_fn(y_pred, y_target)
            return loss
            
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
            return optimizer
    
    modlee_model = ModleeClassifier()

Run the training loop, just for one epoch.

.. code:: ipython3

    with modlee.start_run() as run:
        trainer = modlee.Trainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
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

::

   Run path: /home/ubuntu/projects/modlee_pypi/examples/mlruns/0/7a47086681324d0e924f9076a1262de9/artifacts/model_graph.py
   Saved artifacts: ['transforms.txt', 'model_graph.py', 'model_graph.txt', 'model_size', 'model', 'cached_vars', 'stats_rep', 'snapshot_1.npy', 'lightning_logs', 'snapshot_0.npy', 'model.py', 'loss_calls.txt', 'model_summary.txt']

We can build the model from the cached ``model_graph.Model`` class and
confirm that we can pass an input through it. Note that this model’s
weights will be uninitialized. To load the model from the last
checkpoint, we can load it directly from the cached ``model.pth``.

.. code:: ipython3

    os.chdir(artifacts_path)
    
    # Rebuilding from the object
    import model_graph
    rebuilt_model = model_graph.Model()
    modlee_model.eval(); rebuilt_model.eval()
    x, y = next(iter(train_loader))
    with torch.no_grad():
        y_original = modlee_model(x)
        y_rebuilt = rebuilt_model(x)
    assert y_original.shape == y_rebuilt.shape
    
    # Reloading from the checkpoint
    reloaded_model = torch.load(os.path.join(artifacts_path, 'model', 'data','model.pth'))
    y_reloaded = reloaded_model(x)
    assert y_original.shape == y_reloaded.shape
    print(f"Original input and output shapes: {x.shape}, {y_original.shape}")
    print(f"Output shapes from module-rebuilt and checkpoint-reloaded models: {y_rebuilt.shape}, {y_reloaded.shape}")

::

   Original input and output shapes: torch.Size([64, 1, 28, 28]), torch.Size([64, 10])
   Output shapes from module-rebuilt and checkpoint-reloaded models: torch.Size([64, 10]), torch.Size([64, 10])

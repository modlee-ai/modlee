Documentation
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

    # Modlee-specific imports
    import modlee
    modlee.init(api_key="modleemichael")   # Replace with your API key
    from modlee.utils import get_fashion_mnist
    from modlee.model import ModleeModel

Load the training data.

.. code:: ipython3

    train_loader, val_loader = get_fashion_mnist()
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

Wrap the model in a ``modlee.model.ModleeModel``. ``ModleeModel``
subclassees ``lightning.pytorch.LightningModule`` and uses the same
design for defining the ``training_step``, ``validation_step``, and
``configure_optimizers`` functions.

.. code:: ipython3

    class LightningClassifier(modlee.model.ModleeModel):
        def __init__(self, num_classes=10, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.classifier = Classifier(num_classes=num_classes)
    
        def forward(self, x):
            return self.classifier(x)
    
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_out = self(x)
            loss = F.cross_entropy(y_out, y)
            return {"loss": loss}
    
        def validation_step(self, val_batch, batch_idx):
            x, y = val_batch
            y_out = self(x)
            loss = F.cross_entropy(y_out, y)
            return loss
            
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
            return optimizer
    
    model = LightningClassifier(num_classes)

Run the training loop, just for one epoch.

.. code:: ipython3

    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )


.. parsed-literal::

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

    Run path: /home/ubuntu/projects/modlee_pypi/examples/mlruns/0/19e6c1aacc8046939225db701aa7dfda
    Saved artifacts: ['model_graph.py', 'model_graph.txt', 'model_size', 'model', 'cached_vars', 'stats_rep', 'snapshot_1.npy', 'snapshot_0.npy', 'model.py', 'loss_calls.txt', 'model_summary.txt']


We can build the model from the cached ``model_graph.Model`` class and
confirm that we can pass an input through it. Note that this model’s
weights will be uninitialized. To load the model from the last
checkpoint, we can load it directly from the cached ``model.pth``.

.. code:: ipython3

    os.chdir(artifacts_path)
    
    # Building from the object
    import model_graph
    rebuilt_model = model_graph.Model()
    x, _ = next(iter(train_loader))
    y_original = model(x)
    y_rebuilt = rebuilt_model(x)
    assert y_original.shape == y_rebuilt.shape
    
    # Loading from the checkpoint
    reloaded_model = torch.load(os.path.join(artifacts_path, 'model', 'data','model.pth'))
    y_reloaded = reloaded_model(x)
    assert y_original.shape == y_reloaded.shape

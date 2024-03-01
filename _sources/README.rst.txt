modlee
======

Introduction
------------

Modlee is a machine learning tool that **documents** experiments for
reproduciblity and **recommends** neural network models suited for a
particular dataset. Modlee bypasses costly machine learning
experimentation by recommending performant models based on prior
experiments. Built on top of MLFlow, Modlee documents traditional
experiment assets (model checkpoints, (hyper)parameters, performance
metrics) and meta-features for
`meta-learning <https://ieeexplore.ieee.org/abstract/document/9428530>`__.
Based on these meta-features from prior experiments, Modlee recommends a
neural network model matched to a new task.

Installation
------------

The Modlee package consists of the client-side interface for the
recommendation API and auxiliary functions for documentation. The
documentation functionality is usable without an account, but the
recommendation functionality requires an API key. To create an account
and receive an API key, `visit our website <https://www.modlee.ai>`__.

PyPI
~~~~

Install ``modlee`` from PyPI:

.. code:: shell

   pip install modlee

Source
~~~~~~

Alternatively, build the package from the source by cloning this
repository and installing from the ``pyproject.toml`` configuration
file:

.. code:: shell

   git clone https://github.com/modlee-ai/modlee
   cd modlee
   pip install .

We have developed the package in Python 3.10. Please raise an issue if
you experience environment errors.

Set API key
~~~~~~~~~~~

Either save your API key to an environment variable:

.. code:: shell

   export MODLEE_API_KEY="my-api-key"

Or pass directly to the ``modlee.init`` function (less recommended):

.. code:: python

   # your_experiment_script.py
   import modlee
   modlee.init(api_key="my-api-key")

Usage
-----

Document
~~~~~~~~

Modlee supports documentation for PyTorch Lightning experiments. Guides
for structuring PyTorch Lightning projects are available
`here <https://lightning.ai/docs/pytorch/stable/starter/converting.html>`__
and
`here <https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09>`__.
Once you have created your experiment script, simply follow the four
“I’s”:

.. code:: python

   # Import the package
   import modlee, lightning

   # Initialize with your API key
   modlee.init(api_key="my-api-key")

   # Inherit the ModleeModel class for your model module,
   # instead of lightning.pytorch.LightningModule
   class MyModel(modlee.model.ModleeModel):
       ...
   model = MyModel()

   # Insert the modlee context manager before training
   with modlee.start_run() as run:
       trainer = lightning.pytorch.trainer(max_epochs=10)
       trainer.fit(
           model=model,
           train_dataloaders=train_dataloader,
       )

Modlee will document experiment assets in a new ``./mlruns/`` directory,
relative to wherever the script was called.

Recommend
~~~~~~~~~

To receive a recommended model based on your dataset:

.. code:: python

   # Import and initialize
   import modlee, lightning
   modlee.init(api_key="my-api-key")

   # Create your dataloaders
   train_dataloader, val_dataloader = your_function_to_get_dataloaders()

   # Create a recommender object and fit to the training dataloader
   recommender = modlee.Recommender()
   recommender.fit(train_dataloader)

   # Get the model from the recommender and train
   model = recommender.model
   with modlee.start_run() as run:
       trainer = lightning.pytorch.Trainer(max_epochs=10)
       trainer.fit(
           model=model,
           train_dataloaders=train_dataloader,
           val_dataloaders=val_dataloader
       )

Examples
--------

Colab
~~~~~

Support
-------

FAQ - link to the section on the site and add some dev-specific ones
here - Discord - Raise an issue - Open PR and read code of conduct

TODO
----

-  [ ] Add logo, links to website and Discord

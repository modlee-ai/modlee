modlee
======

Introduction
------------

Until now, ML R&D has been …
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Isolated**: Even with community resources, critical knowledge often
   remains overlooked or neglected.
-  **Tedious**: Training and evaluating many models is often boring,
   slow and wastes precious time.
-  **Messy**: Months of model experimentation can often feel like
   navigating through an endless maze.

We’ve developed a better way …
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use Modlee to obtain higher-quality solutions more easily in
three simple steps:

-  **1. Connect**: Embrace the opportunity to collaborate effortlessly
   with developers worldwide.
-  **2. Benchmark**: Train a high-quality benchmark ML solution faster &
   easier, regardless of your expertise.
-  **3. Explore**: Improve upon your benchmark solution, and contribute
   with your Modlee collaborators.

.. raw:: html

   <!-- ## Introduction
   Modlee is a machine learning tool that **documents** experiments for reproduciblity and **recommends** neural network models suited for a particular dataset.
   Modlee bypasses costly machine learning experimentation by recommending performant models based on prior experiments.
   Modlee documents traditional experiment assets (model checkpoints, (hyper)parameters, performance metrics) and meta-features for [meta-learning](https://ieeexplore.ieee.org/abstract/document/9428530).
   Based on these meta-features from prior experiments, Modlee recommends a neural network model matched to a new task. -->

Installation
------------

.. raw:: html

   <!-- The Modlee package consists of the client-side interface for the recommendation API and auxiliary functions for documentation.
   The documentation functionality is usable without an account, but the recommendation functionality requires an API key.
   To create an account and receive an API key, [visit our website](https://www.dashboard.modlee.ai). -->

Modlee is a machine learning tool that allows us to benchmark & explore
ML solutions more easily together. You can start building ML with Modlee
today using our python client side interface:

.. raw:: html

   <!-- ### Starter environment

   Here's an example virtual environment for Mac, using `brew` & `virtualenv`, compatiable with Modlee:

   ```
   brew install python@3.10
   python3.10 -m venv venv
   source venv/bin/activate
   ```

   *In this case you may need to use `pip3.10`, depending on your symlinking.* -->

PyPI
~~~~

Install ``modlee`` from PyPI:

.. code:: shell

   pip install modlee

Our package is built on top of Pytorch, Pytorch-lightning, MLFlow & more
to ensure you can continue developing ML with frameworks you’re familiar
with.

We have developed the package in Python 3.10. Please `raise an
issue <https://github.com/modlee-ai/modlee/blob/main/issues>`__ if you
experience environment errors.

Source
~~~~~~

Alternatively, build the package from the source by cloning this
repository and installing from the ``pyproject.toml`` configuration
file:

.. code:: shell

   git clone https://github.com/modlee-ai/modlee
   cd modlee
   pip install .

API key
~~~~~~~

Our Python package seamlessly connects you to your collaborators and
recommends model architectures for your datatasets based on the
experiments shared by your collaborators. At Modlee we’ve built a
powerful flywheel that allows the ML R&D community to work together in
new ways and guide eachother to better ML solutions over time.

To use all of the innovative features of Modlee, you’ll need to `Sign
up <https://www.dashboard.modlee.ai?signUp>`__ and generate an API Key:
*Modlee Purple is free, always.*

Features that require an API key
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Automated experiment collaboration - *Connect*
-  ML model architecture recommendations - *Benchmark*

Features that work without an API key
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Automated local ML experiment documentation - *Explore*

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

Prerequisites
~~~~~~~~~~~~~

Modlee is built on top of `PyTorch
Lightning <https://lightning.ai/docs/pytorch/stable/>`__ and
`MLFlow <https://mlflow.org>`__. While you do not have to be an expert
in either framework to use Modlee, we recommend having at least a
familiarity with machine learning and the experiment pipeline. This
documentation page does not cover the frameworks; we recommend
referencing the
`Lightning <https://lightning.ai/docs/overview/getting-started>`__ and
`MLFlow <https://mlflow.org/docs/latest/index.html>`__ documentation
directly.

Benchmark with model recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Obtain a better benchmark ML solution faster & easier, by using our ML
Model Architecture Recommendations. We recommend model architectures
based on your data modality, task, and data meta-features, and deliver a
trainable model within your python script all within a few lines of
code.

.. code:: python

   # Import and initialize
   import modlee, lightning
   modlee.init(api_key="my-api-key")

   # Create your dataloaders
   train_dataloader, val_dataloader = your_function_to_get_dataloaders()

   # Create a recommender object and fit to the training dataloader
   recommender = modlee.recommender.from_modality_task(
       modality='image',
       task='classification',
       )

   # Fit the recommender to the data meta-features
   recommender.fit(train_dataloader)

Modlee analyzed your dataset locally and extracted meta-features, which
are needed for us to recommend a model architecture for your. Learn more
about how we do this in our
`docs <https://docs.modlee.ai/modules/modlee.data_metafeatures.html>`__.

.. code:: python


   # Get the model from the recommender and train
   model = recommender.model
   with modlee.start_run() as run:
       trainer = modlee.Trainer(max_epochs=10)
       trainer.fit(
           model=model,
           train_dataloaders=train_dataloader,
           val_dataloaders=val_dataloader
       )

We format your recommended model as a ``ModleeModel``, which allows us
to automatically document your experiment locally and share it with your
Modlee collaborators. If you’re signed up for Modlee Purple, that’s the
entire Modlee community! Learn more in our
`docs <https://docs.modlee.ai/modules/modlee.client.html#modlee.client.ModleeClient.post_run>`__.

In training a Modlee recommended model, and sharing key information
about your experiment automatically, you’ve contributed to a powerful
flywheel that will allow the ML R&D community to work together in new
ways and guide eachother to better ML solutions over time.

At the moment we support modalities of ``images`` & ``text``, and tasks
of ``classification``, with more coming soon. Let us know which
modalities and tasks you’d prefer on our
`Discord <https://discord.com/invite/m8YDbWDvrF>`__ in the
package-feature-brainstorming channel. If you’re excited about what
we’re building, help us support your use case by contributing to our
`Github <https://github.com/modlee-ai/modlee/blob/main/docs/CONTRIBUTING.md>`__.

Build your own Modlee model recommendation pipeline and connect your
data today or go through a full recommendation example in more detail:
`Benchmark with model
recommendations <https://docs.modlee.ai/notebooks/recommend.html>`__.

Explore & document
~~~~~~~~~~~~~~~~~~

Using Modlee to obtain a benchmark solution, is an easy way to determine
a great starting point for future model exploration. With Modlee you can
focus more on breaking new ground and less of re-inventing the “ML
experiment” wheel. Define a custom ``ModleeModel``, and share key
information about your ``Automatically Documented Experiments`` to guide
your collaborators towards better solutions, simply through the act of
experimenting. No need to share code, repos, or set up a meeting.

Modlee supports documentation for Lightning experiments. Guides for
structuring PyTorch Lightning projects are available
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
       # Define the model
   model = MyModel()

   # Insert the modlee context manager before training
   with modlee.start_run() as run:
       trainer = modlee.trainer(max_epochs=10)
       trainer.fit(
           model=model,
           train_dataloaders=train_dataloader,
       )

For the sake of illustration, we did not define ``train_dataloader`` &
``MyModel`` above. Read through our `Dataset
guidelines <https://docs.modlee.ai/notebooks/dataset_guidelines.html>`__
& `Model definition
guidelines <https://docs.modlee.ai/notebooks/model_definition_guidelines.html>`__
to learn how to define your own custom datasets and models, while using
Modlee’s ``Automated Experiment Documentation``.

Modlee automatically documents experiment assets in a new ``./mlruns/``
directory, relative to wherever the script was called. Among the assets
is a ``model_graph.py`` module that recreates the model as a graph,
including the ``forward()`` pass:

.. code:: python

   import torch, onnx2torch
   from torch import tensor

   class Model(torch.nn.Module):
       
       def __init__(self):
           ''' Rebuild the model from its base components. '''
           super().__init__()
           setattr(self,'Shape', Shape(**{'start':0,'end':None}))
           setattr(self,'Constant', Constant(**{'value':torch.ones(())*2}))
           setattr(self,'Gather', Gather(**{'axis':0}))
           setattr(self,'Shape_1', Shape(**{'start':0,'end':None}))
           setattr(self,'Constant_1', Constant(**{'value':torch.ones(())*3}))
           setattr(self,'Gather_1', Gather(**{'axis':0}))
           setattr(self,'Conv', torch.nn.modules.conv.Conv2d(**{
               'in_channels':3,
               'out_channels':64,
               'kernel_size':(7, 7),
               'stride':(2, 2),
               'padding':(3, 3),
               'dilation':(1, 1),
               'groups':1,
               'padding_mode':'zeros'}))
           ...
       
       def forward(self, input_1):
           ''' Forward pass an input through the network '''
           shape = self.Shape(input_1)
           constant = self.Constant()
           gather = self.Gather(shape, constant.type(torch.int64))
           shape_1 = self.Shape_1(input_1)
           constant_1 = self.Constant_1()
           gather_1 = self.Gather_1(shape_1, constant_1.type(torch.int64))
           conv = self.Conv(input_1)
           ...

At the moment we support modalities of ``images`` & ``text``, and tasks
of ``classification``, with more coming soon. As with recommendation,
use `Discord <https://discord.com/invite/m8YDbWDvrF>`__ to let us know
which modalities and tasks you’d prefer or help make these changes on
our
`Github <https://github.com/modlee-ai/modlee/blob/main/docs/CONTRIBUTING.md>`__.

Start implementing Modlee to continue your own model experimentation or
go through a full documentation example in more detail to learn more:
`Explore & document <https://docs.modlee.ai/notebooks/document.html>`__.

Roadmap
-------

Open source
~~~~~~~~~~~

-  ☐ Add more data modalities and ML tasks
-  ☐ Client-side features needed by the community

Modlee internal
~~~~~~~~~~~~~~~

We’re working hard on exciting new features to help you build better
together! - *(Modlee Silver & Gold)*

-  ☐ Improvements to model architecture recommendations
-  ☐ Control how you’re connected to Modlee
-  ☐ Query and search your own and collaborators experiments backed up
   to Modlee
-  ☐ Personalized model architecture recommendations based on your own
   and collaborators experiments

Support
-------

Community
~~~~~~~~~

`Join our Discord server <https://discord.com/invite/m8YDbWDvrF>`__ to
discuss & contribute with other Modlee users.

Contributing
~~~~~~~~~~~~

Modlee is designed & maintained by developers passionate about AI
innovation, infrastructure & meta learning. For those like us, we
welcome contributions of any kind: bug reports, feature requests,
tutorials, etc.

Before submitting a pull request, `please read the contribution
guidelines <https://github.com/modlee-ai/modlee/blob/main/docs/CONTRIBUTING.md>`__.

Issues
~~~~~~

If you encounter errors, `please raise an issue in this
repository <https://github.com/modlee-ai/modlee/issues>`__.

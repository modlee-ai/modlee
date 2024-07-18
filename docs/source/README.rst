.. image:: https://github.com/modlee-ai/modlee/raw/main/docs/source/logo_icon.svg
   :width: 50px
   :height: 50px

modlee
======

*Until now, ML R&D has been …*

-  **Isolated**: Even with community resources, critical knowledge often
   remains overlooked or neglected.
-  **Tedious**: Training and evaluating many models is often boring,
   slow and wastes precious time.
-  **Messy**: Months of model experimentation can often feel like
   navigating through an endless maze.

*We’ve developed a better way …* Our Python package seemlessly connects
you to your collaborators and recommends model architectures for your
datatasets based on the experiments shared between you. At Modlee we’ve
built a flywheel that allows the ML R&D community to work together in
new ways and guide each other to better ML solutions.

You can **use Modlee’s Python package** to more easily define high
performing ML models in three simple steps using **our tools**:

-  **1. Connect with our Automated Experiment Collaboration**: Embrace
   the opportunity to collaborate effortlessly with developers &
   colleagues worldwide.
-  **2. Benchmark with our Model Architecture Recommendations**: Train a
   high-quality benchmark ML solution faster and easier, regardless of
   your expertise.
-  **3. Explore with our Automated Experiment Documentation**: Improve
   upon your benchmark solution and contribute with your Modlee
   collaborators.

Installation
------------

We have developed the package in Python 3.10. If you experience
environment errors, `please raise an
issue <https://github.com/modlee-ai/modlee/issues>`__.

PyPI
~~~~

Install ``modlee`` from PyPI:

.. code:: shell

   pip install modlee

.. image:: https://raw.githubusercontent.com/mansiagr4/gifs/c3842c67ca512440eb8c651441fdc364f4e38395/pip_install.gif

Our package is built on top of PyTorch, PyTorch-lightning, MLFlow and
more to ensure you can continue developing ML with frameworks that you
are familiar with.

Source
~~~~~~

Alternatively, you can build the package from the source by cloning this
repository and installing from the ``pyproject.toml`` configuration
file:

.. code:: shell

   git clone https://github.com/modlee-ai/modlee
   cd modlee
   pip install .

.. image:: https://raw.githubusercontent.com/mansiagr4/gifs/main/git_clone.gif

API key
~~~~~~~

To use all of features of Modlee, you’ll need to `Sign
up <https://www.dashboard.modlee.ai?signUp>`__ and generate an API Key:
*Modlee Purple is free, always.*

Features that require an API key
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Automated Experiment Collaboration - *Connect*
-  Modle Architecture Recommendations - *Benchmark*

Features that work without an API key
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Automated Experiment Documentation (local) - *Explore*

Set API key
~~~~~~~~~~~

Either save your API key to an environment variable:

.. code:: shell

   export MODLEE_API_KEY="my-api-key"

.. image:: https://raw.githubusercontent.com/mansiagr4/gifs/main/export%20api.gif

Or pass directly to the ``modlee.init`` function (less recommended):

.. code:: python

   # your_experiment_script.py
   import modlee
   modlee.init(api_key="my-api-key")

.. image:: https://raw.githubusercontent.com/mansiagr4/gifs/main/import%20api.gif

Usage
-----

Prerequisites
~~~~~~~~~~~~~

Modlee is built on top of `PyTorch
Lightning <https://lightning.ai/docs/pytorch/stable/>`__ and
`MLFlow <https://mlflow.org>`__. While you do not have to be an expert
in either framework to use Modlee, we recommend being familiar with
machine learning concepts and techniques. This documentation page does
not cover the frameworks; we recommend referencing the
`Lightning <https://lightning.ai/docs/overview/getting-started>`__ and
`MLFlow <https://mlflow.org/docs/latest/index.html>`__ documentation
directly.

Model Architecture Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Obtain a better benchmark ML solution faster and easier, by using our ML
Model Architecture Recommendations. We recommend model architectures
based on your data modality, task, and data meta-features, and deliver a
trainable model within your python script, all within a few lines of
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
are needed for us to recommend a model architecture. Learn more about
how we do this in our
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

We formatted your recommended model as a ``ModleeModel``, which allows
us to automatically document your experiment locally and share it with
your Modlee collaborators. If you’re signed up for Modlee Purple, that’s
the entire Modlee community! Learn more in our
`docs <https://docs.modlee.ai/modules/modlee.client.html#modlee.client.ModleeClient.post_run>`__.

In training a Modlee recommended model, and sharing key information
about your experiment automatically, you’ve contributed to a powerful
flywheel that will allow the ML R&D community to work together in new
ways and guide one another to better ML solutions over time.

Supported use cases
^^^^^^^^^^^^^^^^^^^

At the moment we support modalities of ``images`` and ``text``, and
tasks of ``classification``, with more coming soon. Let us know which
modalities and tasks you’d prefer on our
`Discord <https://discord.com/invite/m8YDbWDvrF>`__ in the
package-feature-brainstorming channel. If you’re excited about what
we’re building, help us support your use case by contributing to our
`Github <https://github.com/modlee-ai/modlee/blob/main/docs/CONTRIBUTING.md>`__.

Next steps
^^^^^^^^^^

Build your own Modlee model recommendation pipeline and connect your
data today or go through a full recommendation example in more detail:
`Benchmark with model
recommendations <https://docs.modlee.ai/notebooks/recommend.html>`__.

Automated Experiment Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using Modlee to obtain a benchmark solution is an easy way to determine
a great starting point for future model exploration. To explore further
define a custom ``ModleeModel``, which will automatically share key
information about your experiment and help guide your collaborators
towards better solutions. No need to share code, repos, or set up a
meeting.

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

For the sake of illustration, we did not define ``train_dataloader`` and
``MyModel`` above. Read through our `Dataset
guidelines <https://docs.modlee.ai/notebooks/dataset_guidelines.html>`__
and `Model definition
guidelines <https://docs.modlee.ai/notebooks/model_definition_guidelines.html>`__
to learn our guidelines for creating custom datasets and models, to
ensure your experiment is documented correctly.

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

.. _supported-use-cases-1:

Supported use cases
^^^^^^^^^^^^^^^^^^^

At the moment we support modalities of ``images`` and ``text``, and
tasks of ``classification``, with more coming soon. As with
recommendation, use `Discord <https://discord.com/invite/m8YDbWDvrF>`__
to let us know which modalities and tasks you’d prefer or help make
these changes on our
`Github <https://github.com/modlee-ai/modlee/blob/main/docs/CONTRIBUTING.md>`__.

.. _next-steps-1:

Next steps
^^^^^^^^^^

Start implementing Modlee to continue your own model experimentation or
go through a full documentation example in more detail to learn more:
`Explore and
document <https://docs.modlee.ai/notebooks/document.html>`__.

Roadmap
-------

Open source
~~~~~~~~~~~

-  ☐ Add more data modalities and ML tasks: a great way to
-  ☐ Client-side features needed by the community:

Modlee internal
~~~~~~~~~~~~~~~

We’re working hard on exciting new features to help you build better
together! - *(Modlee Silver and Gold)*

-  ☐ Improvements to model architecture recommendations
-  ☐ Control how you’re connected to Modlee
-  ☐ Query and search your own and collaborators experiments backed up
   to Modlee
-  ☐ Personalized model architecture recommendations, based on your own
   and collaborators experiments

Support
-------

Community
~~~~~~~~~

`Join our Discord server <https://discord.com/invite/m8YDbWDvrF>`__ to
discuss and contribute with other Modlee users.

Contributing
~~~~~~~~~~~~

Modlee is designed and maintained by developers passionate about AI
innovation, infrastructure and meta learning. For those like us, we
welcome contributions of any kind: bug reports, feature requests,
tutorials, etc.

Before submitting a pull request, `please read the contribution
guidelines <https://github.com/modlee-ai/modlee/blob/main/docs/CONTRIBUTING.md>`__.

Issues
~~~~~~

If you encounter errors, `please raise an issue in this
repository <https://github.com/modlee-ai/modlee/issues>`__.

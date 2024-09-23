|image1|

Quickstart with Modlee
======================

Welcome to Modlee! This guide will help you get up and running quickly
with Modlee, whether you’re new to machine learning or just new to our
platform.

Account Setup
-------------

`Sign up <https://www.dashboard.modlee.ai?signUp>`__ for a Modlee
account if you haven’t already or `sign
in <https://www.dashboard.modlee.ai/?state=signIn>`__ to access your
account.

--------------

Python Setup
------------

1. Install Python
~~~~~~~~~~~~~~~~~

To use the Modlee package, you’ll need to have Python installed on your
computer. To download Python, visit the `official Python
website <https://www.python.org/downloads/>`__ and download the latest
version. To use the Modlee package, you need at least Python 3.10 or
newer. If this is your first time downloading Python, please refer to
the `official Python installation guide for
beginners <https://wiki.python.org/moin/BeginnersGuide/Download>`__.

2. Set Up a Virtual Environment (Optional but Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a virtual environment helps to manage project-specific
dependencies and avoid conflicts between different projects.

-  **Creating a Virtual Environment**: To create a virtual environment,
   open your terminal or command prompt and navigate to your project
   directory. Run the following command:

   .. code:: bash

      python -m venv myenv

   This command creates a directory named myenv (you can choose any
   name) in your project folder, containing a copy of the Python
   interpreter and a fresh set of libraries.

-  **Activate the Virtual Environment**: Once the virtual environment is
   created, you need to activate it.

   -  On Windows:

      .. code:: bash

         myenv\Scripts\activate

   -  On macOS/Linux:

      .. code:: bash

         source myenv/bin/activate

--------------

Install the Modlee Package
--------------------------

PyPI
~~~~

Once you have Python installed and activated a virtual environment, you
can install the Modlee package from PyPI. Run this command from the
terminal/command line:

.. code:: bash

   pip install modlee

|image2|

Our package is built on top of PyTorch, PyTorch-lightning, MLFlow, and
more to ensure you can continue developing ML with frameworks that you
are familiar with.

Source
~~~~~~

Alternatively, you can build the package from the source by cloning this
repository and installing it from the ``pyproject.toml`` configuration
file:

.. code:: shell

   git clone https://github.com/modlee-ai/modlee
   cd modlee
   pip install .

|image3|

--------------

Set API Key
-----------

Navigate to the dashboard and generate an API key. Either save your API
key to an environment variable:

.. code:: shell

   export MODLEE_API_KEY="my-api-key"

|image4|

Or pass directly to the ``modlee.init`` function (less recommended):

.. code:: python

   # your_experiment_script.py
   import modlee
   modlee.init(api_key="my-api-key")

|image5|

--------------

How to Use Modlee - Quick Example
---------------------------------

Get started with Modlee by following these steps to set up and train a
model using our recommender system. This guide will walk you through the
process with a simple end-to-end example.

.. code:: shell


   !pip install --upgrade numpy modlee lightning torch torchvision

.. code:: python


   import modlee
   import lightning as pl
   import torch
   import torchvision
   from torch.utils.data import DataLoader
   from torchvision import transforms

   # Initialize Modlee with your API key
   modlee.init(api_key="you-api-key")

   transform = transforms.Compose([
       transforms.Grayscale(num_output_channels=1), 
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])

   # Load the Fashion MNIST dataset
   train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
   test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

   # Create DataLoaders for training and testing
   training_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

   # Create a recommender for image classification tasks
   recommender = modlee.recommender.from_modality_task(modality='image', task='classification')

   # Fit the recommender with the training DataLoader
   recommender.fit(training_loader)

   # Retrieve the recommended model from the recommender
   modlee_model = recommender.model
   print(f"\nRecommended model: \n{modlee_model}")

   # Train the recommended model
   with modlee.start_run() as run:
       trainer = pl.Trainer(max_epochs=1)
       trainer.fit(
           model=modlee_model,
           train_dataloaders=training_loader,
           val_dataloaders=test_loader
       )

You should see a recommended model as an output. If you are running into
issues, please refer to our `Troubleshooting
Page <https://docs.modlee.ai/notebooks/troubleshooting.html>`__ for more
help.

--------------

Supported Use Cases
-------------------

At the moment we support modalities of ``images`` and ``text``, and
tasks of ``classification``, with more coming soon. As with
recommendation, use `Discord <https://discord.com/invite/m8YDbWDvrF>`__
to let us know which modalities and tasks you’d prefer or help make
these changes on our
`GitHub <https://github.com/modlee-ai/modlee/blob/main/docs/CONTRIBUTING.md>`__.

Recommended Next Steps
----------------------

To further develop your expertise, explore the following:

1. `Visit the Projects
   Page <https://docs.modlee.ai/notebooks/tutorials.html>`__: Browse our
   projects page for guided examples and step-by-step instructions.
   These projects are designed to help you get hands-on experience with
   Modlee and apply it effectively.

2. `Dive into the
   Guides <https://docs.modlee.ai/notebooks/guides.html>`__: Explore
   Modlee’s in-depth guides to discover advanced features and
   capabilities. These resources offer detailed instructions and
   practical examples to enhance your proficiency.

3. `Explore the
   Examples <https://docs.modlee.ai/notebooks/recommend.html>`__: Check
   out our collection of examples to see how Modlee is used across
   various tasks. These examples can spark ideas and show you how to
   implement Modlee in your projects.

4. `Join the Community <https://docs.modlee.ai/support.html>`__:
   Participate in discussions and forums to connect with other users,
   seek advice, and share your experiences. Engaging with the community
   can provide additional support and insights.

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg
.. |image2| image:: https://raw.githubusercontent.com/mansiagr4/gifs/main/trimmed_pip_install.gif
.. |image3| image:: https://raw.githubusercontent.com/mansiagr4/gifs/main/trimmed_git_clone.gif
.. |image4| image:: https://raw.githubusercontent.com/mansiagr4/gifs/main/export%20api.gif
.. |image5| image:: https://raw.githubusercontent.com/mansiagr4/gifs/main/import%20api.gif

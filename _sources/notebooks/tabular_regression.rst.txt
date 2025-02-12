|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Tabular Regression
==================

In this tutorial, we will walk through the process of building a tabular
regression model using the Modlee package. We will utilize the
``California Housing`` dataset, which contains information about housing
prices in California, to predict house prices based on various features.

|Open in Kaggle|

First, we will import the the necessary libraries and set up the
environment.

.. code:: python

   import torch
   import os
   import modlee
   import lightning.pytorch as pl
   from torch.utils.data import DataLoader, TensorDataset
   from sklearn.datasets import fetch_california_housing
   from sklearn.model_selection import train_test_split
   import pytest
   from utils import check_artifacts

Now, we will set up the ``modlee`` API key and initialize the ``modlee``
package. You can access your ``modlee`` API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__.

Replace ``replace-with-your-api-key`` with your API key.

.. code:: python

   modlee.init(api_key="replace-with-your-api-key")

Now, we will load the ``California Housing`` dataset, which we will use
for our regression task. This dataset is readily available through the
``fetch_california_housing`` function from the ``sklearn.datasets``
module. We will convert the features and target values into ``PyTorch``
tensors for compatibility with our model.

.. code:: python

   def load_california_housing_data():
       # Fetch the California housing dataset
       data = fetch_california_housing()
       X, y = data.data, data.target  # Separate features (X) and target (y)
       
       # Convert features and target to PyTorch tensors for compatibility
       X = torch.tensor(X, dtype=torch.float32)
       y = torch.tensor(y, dtype=torch.float32)
       
       return X, y

   # Load the data
   X, y = load_california_housing_data()

Once we have the data, the next step is to split it into training and
testing sets. This allows us to train our model on one set of data and
validate its performance on another.

.. code:: python

   # Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

After splitting the data, we need to create ``TensorDataset`` objects
for our training and testing sets. This will facilitate easy loading of
the data during the training process. We create ``DataLoader`` objects
for both training and testing datasets.

.. code:: python

   # Create TensorDataset objects for training and testing data
   train_dataset = TensorDataset(X_train, y_train)
   test_dataset = TensorDataset(X_test, y_test)

   # Create DataLoader objects for batching and shuffling
   train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

Now it’s time to define our regression model using Modlee’s framework.
We will create a class called ``TabularRegression`` that inherits from
``modlee.model.TabularRegressionModleeModel``. This class will define
our neural network architecture and the training process.

.. code:: python

   class TabularRegression(modlee.model.TabularRegressionModleeModel):
       def __init__(self, input_dim):
           super().__init__()  # Initialize the parent class
           # Define the neural network architecture
           self.model = torch.nn.Sequential(
               torch.nn.Linear(input_dim, 128),  # First layer with 128 neurons
               torch.nn.ReLU(),  # ReLU activation function
               torch.nn.Linear(128, 64),  # Second layer with 64 neurons
               torch.nn.ReLU(),  # ReLU activation function
               torch.nn.Linear(64, 1)  # Output layer predicting a single value
           )
           self.loss_fn = torch.nn.MSELoss()  # Mean Squared Error loss function

       def forward(self, x):
           return self.model(x)  # Forward pass through the model

       def training_step(self, batch):
           x, y = batch  # Unpack the batch
           preds = self.forward(x).squeeze()  # Get predictions from the model
           loss = self.loss_fn(preds, y)  # Compute loss
           return loss  # Return the loss

       def validation_step(self, batch):
           x, y = batch  # Unpack the batch
           preds = self.forward(x).squeeze()  # Get predictions from the model
           loss = self.loss_fn(preds, y)  # Compute loss
           return loss  # Return the loss

       def configure_optimizers(self):
           return torch.optim.Adam(self.parameters(), lr=1e-3)  # Optimizer configuration

   modlee_model = TabularRegression(input_dim=X_train.shape[1])

With the model defined, we can proceed to train it. We will use the
``pl.Trainer`` from ``PyTorch Lightning``, which simplifies the training
process. We will specify the number of epochs and how often to log
training progress.

.. code:: python

   with modlee.start_run() as run:  # Start a training run
       trainer = pl.Trainer(max_epochs=1)  # Set up the trainer
       trainer.fit(  # Start training the model
           model=modlee_model,
           train_dataloaders=train_dataloader,
           val_dataloaders=test_dataloader  # Use test data for validation
       )

After training, we inspect the artifacts saved by Modlee, including the
model graph and various statistics. With Modlee, your training assets
are automatically saved, preserving valuable insights for future
reference and collaboration.

.. code:: python

   last_run_path = modlee.last_run_path()
   print(f"Run path: {last_run_path}")
   artifacts_path = os.path.join(last_run_path, 'artifacts')
   artifacts = sorted(os.listdir(artifacts_path))
   print(f"Saved artifacts: {artifacts}")

.. |Open in Kaggle| image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/code/modlee/tabular-regression

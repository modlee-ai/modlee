|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Time Series Classification
==========================

In this tutorial, we will walk through the process of building a
multivariate time series classification model using Modlee and
``PyTorch``.

Time series classification is a task where models predict categorical
labels based on sequential input data. We will use a dataset that
contains time series data representing different car outlines extracted
from video footage.

**Note**: Currently, Modlee does not support recurrent LSTM operations.
Instead, we will focus on non-recurrent models suited for time series
data, such as convolutional neural networks (CNNs) and transformers,
which can effectively capture sequential patterns without requiring
recurrent layers.

|Open in Kaggle|

First, we will import the the necessary libraries and set up the
environment.

.. code:: python

   import torch
   import os
   import modlee
   import lightning.pytorch as pl
   from torch.utils.data import DataLoader, TensorDataset
   import pandas as pd

Now, we will set up the ``modlee`` API key and initialize the ``modlee``
package. You can access your ``modlee`` API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__.

Replace ``replace-with-your-api-key`` with your API key.

.. code:: python

   modlee.init(api_key="replace-with-your-api-key")

The dataset we will use consists of time series data that represent
outlines of four different types of cars (sedan, pickup, minivan, SUV)
extracted from traffic videos using motion information. Each vehicle is
mapped onto a 1-D series, where each series captures the vehicle’s
outline. The objective is to classify these series into one of the four
classes.

For this example, we will manually download the dataset from Kaggle and
upload it to the environment. Visit the `Time Series Classification
dataset
page <https://www.kaggle.com/datasets/modlee/time-series-classification-data>`__
on Kaggle and click the **Download** button to save the dataset to your
local machine.

Copy the path to the donwloaded files, which will be used later.

To load the data, we create a function that reads the files and
processes them into ``PyTorch`` tensors. Each time series entry has
features representing the outline of a vehicle, with the first column in
the dataset being the target label.

.. code:: python

   def load_car_from_txt(file_path):
       # Load the dataset with space as the delimiter and no header
       data = pd.read_csv(file_path, delim_whitespace=True, header=None)
       y = data.iloc[:, 0].values  # The first column represents the target (car type)
       X = data.iloc[:, 1:].values  # The rest of the columns represent the time series features
       
       # Convert the features and labels to PyTorch tensors
       X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Add a dimension for input size
       y = torch.tensor(y, dtype=torch.long)  # Ensure labels are in long tensor format for classification
       return X, y

   # Load the training data
   train_file_path = 'path-to-Car_TRAIN.txt'
   X_train, y_train = load_car_from_txt(train_file_path)

   # Load the test data
   test_file_path = 'path-to-Car_TEST.txt'
   X_test, y_test = load_car_from_txt(test_file_path)

After loading the data, we create ``PyTorch TensorDataset`` and
``DataLoader`` objects to facilitate data handling during training and
validation.

.. code:: python

   # Create PyTorch TensorDatasets
   train_dataset = TensorDataset(X_train, y_train)
   test_dataset = TensorDataset(X_test, y_test)

   # Create DataLoaders for training and testing
   train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
   test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

We define a Transformer-based neural network for multivariate time
series classification. The model includes:

-  A ``TransformerEncoder`` layer to capture sequence dependencies.
-  A fully connected ``(fc)`` layer that maps the encoder output to
   class labels.
-  Cross-entropy loss for training, optimized with the Adam optimizer.

.. code:: python

   class TransformerTimeSeriesClassifier(modlee.model.TimeseriesClassificationModleeModel):
       def __init__(self, input_dim, seq_length, num_classes, num_heads=1, hidden_dim=64):
           super().__init__()
           # Define a Transformer encoder layer with specified input dimension and number of attention heads
           self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
           # Stack Transformer encoder layers to create a Transformer encoder
           self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=2)
           # Fully connected layer to map encoded features to class scores
           self.fc = torch.nn.Linear(input_dim * seq_length, num_classes)
           # Set the loss function to CrossEntropyLoss for multi-class classification
           self.loss_fn = torch.nn.CrossEntropyLoss()

       def forward(self, x):
           # Pass input through the Transformer encoder to capture dependencies
           x = self.transformer_encoder(x)
           # Flatten the output and pass it through the fully connected layer for class prediction
           x = x.view(x.size(0), -1)
           x = self.fc(x)
           return x 

       def training_step(self, batch):
           # Get input data and target labels from batch
           x, y = batch
           # Forward pass to generate predictions
           preds = self.forward(x)
           # Calculate loss using the specified loss function
           loss = self.loss_fn(preds, y)
           return loss

       def validation_step(self, batch):
           # Get input data and target labels from batch
           x, y = batch
           # Forward pass to generate predictions
           preds = self.forward(x)
           # Calculate validation loss
           loss = self.loss_fn(preds, y)
           return loss

       def configure_optimizers(self):
           # Use the Adam optimizer with a learning rate of 1e-3 for optimization
           return torch.optim.Adam(self.parameters(), lr=1e-3)

   # Instantiate the model with specified parameters
   modlee_model = TransformerTimeSeriesClassifier(input_dim=1, seq_length=577, num_classes=4)

To train the model, we use ``PyTorch Lightning's Trainer`` class, which
simplifies the training loop.

.. code:: python

   # Start a Modlee run for tracking
   with modlee.start_run() as run:
       trainer = pl.Trainer(max_epochs=1)
       trainer.fit(
           model=modlee_model,
           train_dataloaders=train_dataloader,
           val_dataloaders=test_dataloader
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
   :target: https://www.kaggle.com/code/modlee/time-series-classification

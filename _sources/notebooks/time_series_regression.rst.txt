|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Time Series Regression
======================

In this tutorial, we will guide you through the process of implementing
a time series regression model using the Modlee framework along with
``PyTorch``.

The goal is to predict power consumption based on various environmental
factors, such as temperature, humidity, wind speed, and solar radiation.

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
   import pytest
   import pandas as pd
   from sklearn.model_selection import train_test_split

Now, we will set up the ``modlee`` API key and initialize the ``modlee``
package. You can access your ``modlee`` API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__.

Replace ``replace-with-your-api-key`` with your API key.

.. code:: python

   modlee.init(api_key="replace-with-your-api-key")

The dataset used in this tutorial includes hourly time series data that
links environmental conditions to power consumption across three zones.
Each record contains a timestamp, temperature, humidity, wind speed, and
measures of solar radiation, alongside the power consumption (in watts)
for each zone.

This data allows for the exploration of relationships between weather
patterns and energy usage, aiding in the development of predictive
models.

For this example, we will manually download the dataset from Kaggle and
upload it to the environment. Visit the `Time Series Regression dataset
page <https://www.kaggle.com/datasets/modlee/time-series-regression-data>`__
on Kaggle and click the **Download** button to save the dataset to your
local machine.

Copy the path to the donwloaded files, which will be used later.

Next, we need to load the power consumption dataset. This dataset
contains various features related to environmental conditions and their
corresponding power consumption values. The
``load_power_consumption_data`` function is designed to read the CSV
file, process the data, and create time series sequences.

We then select the relevant features from the dataset for our input
variables, ``X``, which include temperature, humidity, wind speed, and
solar radiation values. The output variable, ``y``, is calculated as the
mean power consumption across three different zones.

.. code:: python

   # Function to load the power consumption dataset and prepare it for training
   def load_power_consumption_data(file_path, seq_length):
       # Load the dataset from the specified CSV file
       data = pd.read_csv(file_path)
       # Convert the 'Datetime' column to datetime objects
       data['Datetime'] = pd.to_datetime(data['Datetime'])
       # Set the 'Datetime' column as the index for the DataFrame
       data.set_index('Datetime', inplace=True)
       
       # Extract relevant features for prediction and target variable
       X = data[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']].values
       # Calculate the average power consumption across the three zones as the target variable
       y = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']].mean(axis=1).values  
       # Convert features and target to PyTorch tensors
       X = torch.tensor(X, dtype=torch.float32)
       y = torch.tensor(y, dtype=torch.float32)

       # Create sequences of the specified length for input features
       num_samples = X.shape[0] - seq_length + 1
       X_seq = torch.stack([X[i:i + seq_length] for i in range(num_samples)])
       y_seq = y[seq_length - 1:]  # Align target variable with sequences

       return X_seq, y_seq

Once we have the preprocessed data, we proceed to create ``PyTorch``
datasets and ``DataLoaders``.

Here, we load the power consumption data from the specified CSV file. We
create a ``TensorDataset`` to hold the features and labels. To split the
dataset into training and validation sets, we use the
``train_test_split`` function from ``sklearn``.

.. code:: python

   # Define the path to the dataset
   file_path = 'path-to-powerconsumption.csv'
   # Load the power consumption data with a specified sequence length
   X, y = load_power_consumption_data(file_path, 20)

   # Create a TensorDataset for the training data
   dataset = TensorDataset(X, y)
   # Split dataset indices into training and validation sets
   train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

   # Create training and validation datasets
   train_dataset = TensorDataset(X[train_indices], y[train_indices])
   val_dataset = TensorDataset(X[val_indices], y[val_indices])

   # Create DataLoader for batch processing during training
   train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

We will now define a multivariate time series regression model by
creating a class that inherits from
``modlee.model.TimeseriesRegressionModleeModel``. This class uses a
Transformer-based architecture to predict a continuous value.

We initialize a ``TransformerEncoder`` with multi-head attention to
process sequential dependencies.

.. code:: python

   class TransformerTimeSeriesRegressor(modlee.model.TimeseriesRegressionModleeModel):
       def __init__(self, input_dim, seq_length, num_heads=1, hidden_dim=64):
           super().__init__()
           # Initialize a Transformer encoder layer with specified input dimensions and heads
           self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
           # Stack encoder layers to form the Transformer encoder
           self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=2)
           # Define a fully connected layer to map encoded features to a single output value
           self.fc = torch.nn.Linear(input_dim * seq_length, 1)
           # Set the loss function to mean squared error for regression tasks
           self.loss_fn = torch.nn.MSELoss()

       def forward(self, x):
           # Pass input through the Transformer encoder
           x = self.transformer_encoder(x)
           # Flatten the output and pass it through the fully connected layer
           x = x.view(x.size(0), -1)
           x = self.fc(x)
           return x 

       def training_step(self, batch):
           # Get input and target from batch
           x, y = batch
           # Generate predictions and compute loss
           preds = self.forward(x)
           loss = self.loss_fn(preds, y)
           return loss

       def validation_step(self, batch):
           # Get input and target from batch
           x, y = batch
           # Generate predictions and compute loss
           preds = self.forward(x)
           loss = self.loss_fn(preds, y)
           return loss

       def configure_optimizers(self):
           # Use the Adam optimizer with a learning rate of 1e-3
           return torch.optim.Adam(self.parameters(), lr=1e-3)

   model = TransformerTimeSeriesRegressor(input_dim=5, seq_length=20)

With our model defined, we can now train it using the
``PyTorch Lightning Trainer``. This trainer simplifies the training
process by managing the training loops and logging.

.. code:: python

   # Start a training run with Modlee
   with modlee.start_run() as run:
       trainer = pl.Trainer(max_epochs=1)  # Set up the trainer
       trainer.fit(
           model=model,
           train_dataloaders=train_dataloader,  # Load training data
           val_dataloaders=val_dataloader  # Load validation data
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
   :target: https://www.kaggle.com/code/modlee/time-series-regression

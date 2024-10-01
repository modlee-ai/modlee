|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Time Series Forecasting
=======================

This examples uses the ``modlee`` package for time series forecasting.
We’ll use the ``Air Passengers`` dataset to show you how to:

1. Prepare the Data: Load and preprocess the dataset, including scaling
   and splitting into training and test sets.
2. Use Modlee for Model Training: Train a model using Modlee’s
   framework.
3. Evaluate Model: Assess the performance of the trained model on the
   test data.

Note: currently `TimeseriesForecastingModleeModel` does not support PyTorch Recurrent Layers.

|Open in Kaggle|

First, we will import the the necessary libraries and set up the
environment.

.. code:: python

   import torch
   import os
   import modlee
   import lightning.pytorch as pl
   from torch.utils.data import DataLoader, TensorDataset
   from sklearn.preprocessing import MinMaxScaler
   from sklearn.model_selection import train_test_split
   import numpy as np
   import seaborn as sns

Now, we will set up the ``modlee`` API key and initialize the ``modlee``
package. You can access your ``modlee`` API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__.

Replace ``replace-with-your-api-key`` with your API key.

.. code:: python

   os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
   modlee.init(api_key=os.environ['MODLEE_API_KEY'])

Next, we will prepare and load our data. We will use the
``Air Passengers`` dataset, which contains the monthly number of airline
passengers over several years. We can load this dataset using the
``seaborn`` library.

.. code:: python

   data = sns.load_dataset('flights')

Now, we will prepare our data for training. We create a function called
``prepare_air_passenger_data`` to handle this process.

.. code:: python

   def prepare_air_passenger_data(data, seq_length):
       # Convert the 'passengers' column to a float array
       passenger_data = data['passengers'].values.astype(float)
       
       # Scale the data to the range [0, 1]
       scaler = MinMaxScaler(feature_range=(0, 1))
       passenger_data = scaler.fit_transform(passenger_data.reshape(-1, 1))

       X, y = [], []
       # Create sequences of length seq_length for forecasting
       for i in range(len(passenger_data) - seq_length):
           X.append(passenger_data[i:i + seq_length])  # Append the sequence
           y.append(passenger_data[i + seq_length])     # Append the target value
       
       X = np.array(X)
       y = np.array(y)

       return X, y, scaler  # Return the sequences, target values, and scaler

We will call the ``prepare_air_passenger_data`` function and also split
the dataset into training and testing sets.

.. code:: python

   # Set the sequence length to 12, indicating we will use the past 12 months of data
   seq_length = 12 

   # Prepare the data by calling the function to get the sequences and target values
   X, y, scaler = prepare_air_passenger_data(data, seq_length)

   # Reshape X to match the input shape required for the MLP: (batch_size, seq_length, input_dim)
   X = X.reshape(-1, seq_length, 1)  
   y = y.reshape(-1, 1)  

   # Split the data into training and test sets (80% training, 20% testing)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Convert the training and test sets to PyTorch tensors
   X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train) 
   X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)    

Now, we can create our dataloaders for both the training and testing
datasets. We will use the ``TensorDataset`` class from ``PyTorch`` to
create datasets from the training and testing tensors.

.. code:: python

   # Create TensorDataset for training and test data
   train_dataset = TensorDataset(X_train, y_train)
   test_dataset = TensorDataset(X_test, y_test)

   # Create DataLoader instances for training and validation
   train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

After preparing the data, we need to define our model. We will create a
simple Multi-Layer Perceptron (MLP) model for time series forecasting
using Modlee’s framework.

.. code:: python

   # Define the MLP model for time series forecasting, inheriting from Modlee's model
   class TimeSeriesForecasterMLP(modlee.model.TimeseriesForecastingModleeModel):
       def __init__(self, input_dim, seq_length, hidden_dim=64):
           super().__init__()
           self.seq_length = seq_length  # Store the sequence length
           self.hidden_dim = hidden_dim    # Set the number of hidden units

           # Define the layers of the MLP
           self.fc1 = torch.nn.Linear(input_dim * seq_length, hidden_dim)  # First hidden layer
           self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
           self.fc3 = torch.nn.Linear(hidden_dim, 1)  # Output layer

           # Define the loss function (Mean Squared Error)
           self.loss_fn = torch.nn.MSELoss()

       def forward(self, x):
           x = x.view(x.size(0), -1)  # Flatten the input for MLP
           x = torch.relu(self.fc1(x))  # Apply ReLU activation after first layer
           x = torch.relu(self.fc2(x))  # Apply ReLU activation after second layer
           predictions = self.fc3(x)  # Generate predictions from the output layer
           return predictions

       def training_step(self, batch):
           x, y = batch  # Unpack the input features and targets
           preds = self.forward(x)  # Forward pass to get predictions
           loss = self.loss_fn(preds, y)  # Calculate loss
           return loss

       def validation_step(self, batch):
           x, y = batch  # Unpack the input features and targets
           preds = self.forward(x)  # Forward pass to get predictions
           loss = self.loss_fn(preds, y)  # Calculate validation loss
           return loss

       def configure_optimizers(self):
           # Use Adam optimizer for model parameters
           return torch.optim.Adam(self.parameters(), lr=1e-3)

Now, we can proceed to train our model using the Modlee package. We
create an instance of our ``TimeSeriesForecasterMLP`` model and then set
up the training loop using the ``Trainer`` class from
``PyTorch Lightning``.

.. code:: python

   input_dim = 1  # We have one feature (number of passengers)

   # Initialize the Modlee model
   model = TimeSeriesForecasterMLP(input_dim, seq_length)

   # Train the model using PyTorch Lightning
   with modlee.start_run() as run:
       trainer = pl.Trainer(max_epochs=1)
       trainer.fit(
           model=model,
           train_dataloaders=train_dataloader,
           val_dataloaders=test_dataloader
       )

Finally, we inspect the artifacts saved by Modlee, including the model
graph and various statistics. With Modlee, your training assets are
automatically saved, preserving valuable insights for future reference
and collaboration.

.. code:: python

   last_run_path = modlee.last_run_path()
   print(f"Run path: {last_run_path}")
   artifacts_path = os.path.join(last_run_path, 'artifacts')
   artifacts = sorted(os.listdir(artifacts_path))
   print(f"Saved artifacts: {artifacts}")

.. |Open in Kaggle| image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/code/modlee/modlee-time-series-forecasting

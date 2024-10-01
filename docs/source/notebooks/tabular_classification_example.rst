|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Tabular Classification
======================

This examples uses the ``modlee`` package for tabular data
classification. We’ll use a diabetes dataset to show you how to:

1. Prepare the data.
2. Use ``modlee`` for model training.
3. Implement and train a custom model.
4. Evaluate the model.

|Open in Kaggle|

First, we will import the the necessary libraries and set up the
environment.

.. code:: python

   import pandas as pd
   from sklearn.preprocessing import StandardScaler
   import torch
   import os
   import modlee
   import lightning.pytorch as pl
   from torch.utils.data import DataLoader, TensorDataset, random_split
   from sklearn.model_selection import train_test_split
   import ssl
   ssl._create_default_https_context = ssl._create_unverified_context

Now, we will set up the ``modlee`` API key and initialize the ``modlee``
package. You can access your ``modlee`` API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__.

Replace ``replace-with-your-api-key`` with your API key.

.. code:: python

   os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
   modlee.init(api_key=os.environ['MODLEE_API_KEY'])

Now, we will prepare our data. For this example, we will manually
download the diabetes dataset from Kaggle and upload it to the
environment.

Visit the `Diabetes CSV dataset
page <https://www.kaggle.com/datasets/saurabh00007/diabetescsv>`__ on
Kaggle and click the **Download** button to save the dataset
``diabetes.csv`` to your local machine.

Copy the path to that donwloaded file, which will be used later.

Define a custom dataset class ``TabularDataset`` for handling our
tabular data.

.. code:: python

   class TabularDataset(TensorDataset):
       def __init__(self, data, target):
           self.data = torch.tensor(data, dtype=torch.float32)  # Convert features to tensors
           self.target = torch.tensor(target, dtype=torch.long) # Convert labels to long integers for classification

       def __len__(self):
           return len(self.data) # Return the size of the dataset

       def __getitem__(self, idx):
           return self.data[idx], self.target[idx] # Return a single sample from the dataset

We can now load and preprocess the data, and also create the
dataloaders.

.. code:: python

   def get_diabetes_dataloaders(batch_size=32, val_split=0.2, shuffle=True):
       dataset_path = "path-to-dataset"
       df = pd.read_csv(dataset_path) # Load the CSV file into a DataFrame
       X = df.drop('Outcome', axis=1).values # Features (X) - drop the target column
       y = df['Outcome'].values # Labels (y) - the target column
       scaler = StandardScaler() # Initialize the scaler for feature scaling
       X_scaled = scaler.fit_transform(X) # Scale the features
       dataset = TabularDataset(X_scaled, y) # Create a TabularDataset instance

       # Split the dataset into training and validation sets
       dataset_size = len(dataset)
       val_size = int(val_split * dataset_size)
       train_size = dataset_size - val_size
       train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

       # Create DataLoader instances for training and validation
       train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
       val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

       return train_dataloader, val_dataloader

   # Generate the DataLoaders
   train_dataloader, val_dataloader = get_diabetes_dataloaders(batch_size=32, val_split=0.2, shuffle=True)

Next, we will define our custom model, which is a simple feedforward
neural network called ``TabularClassifier``. This model will be
integtated with Modlee’s framework.

.. code:: python

   class TabularClassifier(modlee.model.TabularClassificationModleeModel):
       def __init__(self, input_dim, num_classes=2):
           super().__init__()
           self.fc1 = torch.nn.Linear(input_dim, 128)  
           self.dropout1 = torch.nn.AlphaDropout(0.1) 

           self.fc2 = torch.nn.Linear(128, 64)  
           self.dropout2 = torch.nn.AlphaDropout(0.1)  

           self.fc3 = torch.nn.Linear(64, 32) 
           self.dropout3 = torch.nn.AlphaDropout(0.1) 

           self.fc4 = torch.nn.Linear(32, num_classes)  

           self.loss_fn = torch.nn.CrossEntropyLoss()

       def forward(self, x):
           x = torch.selu(self.fc1(x))  
           x = self.dropout1(x) 

           x = torch.selu(self.fc2(x))  
           x = self.dropout2(x)  

           x = torch.selu(self.fc3(x))  
           x = self.dropout3(x)  

           x = self.fc4(x)  
           return x

       def training_step(self, batch, batch_idx):
           x, y_target = batch
           y_pred = self(x)
           loss = self.loss_fn(y_pred, y_target.squeeze())
           return {"loss": loss}

       def validation_step(self, val_batch, batch_idx):
           x, y_target = val_batch
           y_pred = self(x)
           val_loss = self.loss_fn(y_pred, y_target.squeeze()) 
           return {'val_loss': val_loss}

       def configure_optimizers(self):
           optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)  
           return optimizer

Next, we can train and evaluate our model using ``PyTorch Lightning``
for one epoch.

.. code:: python

   # Get the input dimension
   original_train_dataset = train_dataloader.dataset.dataset 
   input_dim = len(original_train_dataset[0][0])
   num_classes = 2  

   # Initialize the Modlee model
   modlee_model = TabularClassifier(input_dim=input_dim, num_classes=num_classes)

   # Train the model using PyTorch Lightning
   with modlee.start_run() as run:
       trainer = pl.Trainer(max_epochs=1)
       trainer.fit(
           model=modlee_model,
           train_dataloaders=train_dataloader,
           val_dataloaders=val_dataloader
       )

Now, we inspect the artifacts saved by Modlee, including the model graph
and various statistics. With Modlee, your training assets are
automatically saved, preserving valuable insights for future reference
and collaboration.

.. code:: python

   import sys

   # Get the path to the last run's saved data
   last_run_path = modlee.last_run_path()
   print(f"Run path: {last_run_path}")

   # Get the path to the saved artifacts
   artifacts_path = os.path.join(last_run_path, 'artifacts')
   artifacts = os.listdir(artifacts_path)
   print(f"Saved artifacts: {artifacts}")

   # Set the artifacts path as an environment variable
   os.environ['ARTIFACTS_PATH'] = artifacts_path

   # Add the artifacts directory to the system path
   sys.path.insert(0, artifacts_path)

.. code:: python

   # Print out the first few lines of the model
   print("Model graph:")

.. code:: shell

   !sed -n -e 1,15p $ARTIFACTS_PATH/model_graph.py
   !echo "        ..."
   !sed -n -e 58,68p $ARTIFACTS_PATH/model_graph.py
   !echo "        ..."

.. code:: python

   # Print the first lines of the data metafeatures
   print("Data metafeatures:")

.. code:: shell

   !head -20 $ARTIFACTS_PATH/stats_rep

.. |Open in Kaggle| image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/code/modlee/modlee-tabular-classification

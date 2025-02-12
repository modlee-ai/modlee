|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Image Regression
================

In this tutorial, we will guide you through building and training a deep
learning model for age estimation from facial images using Modlee and
``PyTorch``.

This project demonstrates handling image data, constructing custom
datasets, defining a neural network for regression, and training it
effectively.

|Open in Kaggle|

First, we will import the the necessary libraries and set up the
environment.

.. code:: python

   import torch
   import os
   import modlee
   import pytest
   import lightning.pytorch as pl
   from torch.utils.data import DataLoader, Dataset
   from torch import nn
   from torchvision import transforms
   from PIL import Image
   import pandas as pd

Now, we will set up the ``modlee`` API key and initialize the ``modlee``
package. You can access your ``modlee`` API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__.

Replace ``replace-with-your-api-key`` with your API key.

.. code:: python

   modlee.init(api_key="replace-with-your-api-key")

The dataset we are using contains facial images labeled into three
distinct age groups: ‘YOUNG’, ‘MIDDLE’, and ‘OLD’. These groups are
mapped to numerical values (0, 1, and 2, respectively) for training
purposes. The dataset is intended for use cases such as biometric
analysis and media content age control, showcasing how deep learning can
power practical applications in facial analysis.

For this example, we will manually download the dataset from Kaggle and
upload it to the environment. Visit the `Face Age Detection dataset
page <https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset>`__
on Kaggle and click the **Download** button to save the dataset to your
local machine.

Copy the path to the donwloaded files, which will be used later.

Define a custom dataset class ``TabularDataset`` for handling our
tabular data.

We create the ``AgeDataset`` class to handle loading and preprocessing
of images and labels from the dataset. The ``AgeDataset`` class inherits
from ``torch.utils.data.Dataset``.

.. code:: python

   AGE_GROUPS = {'YOUNG': 0, 'MIDDLE': 1, 'OLD': 2}  # Mapping age groups to numerical labels

   class AgeDataset(Dataset):
       def __init__(self, image_folder, csv_file, img_size, transform=None):
           self.image_folder = image_folder
           self.data = pd.read_csv(csv_file)  # Read the CSV file containing image paths and labels
           self.img_size = img_size
           self.transform = transform or transforms.Compose([
               transforms.Resize(self.img_size[1:]),  # Resize images to the specified size
               transforms.ToTensor(),  # Convert images to PyTorch tensors
           ])
           # Map the 'Class' column to numerical labels
           if 'Class' in self.data.columns:
               self.data['Class'] = self.data['Class'].map(AGE_GROUPS)

       def __len__(self):
           return len(self.data)

       def __getitem__(self, idx):
           img_path = os.path.join(self.image_folder, self.data.iloc[idx, 0])
           label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

           image = Image.open(img_path).convert('RGB')  # Open the image and convert to RGB
           if self.transform:
               image = self.transform(image)  # Apply transformations

           return image, label

Now, we specify the file paths for the image folder and CSV file, create
an instance of ``AgeDataset`` to load and preprocess the data, and
initialize a ``DataLoader`` to batch and shuffle the data.

.. code:: python

   # Define the paths to the image folder and CSV file
   image_folder = 'path-to-folder'
   csv_file = 'path-to-csv'

   # Initialize the dataset and dataloader
   train_dataset = AgeDataset(image_folder, csv_file, img_size=(3, 32, 32))
   train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

Next, we define the regression model using the ``ModleeImageRegression``
class. The ``ModleeImageRegression`` class extends the
``modlee.model.ImageRegressionModleeModel`` class, setting up a simple
neural network for regression tasks.

.. code:: python

   class ModleeImageRegression(modlee.model.ImageRegressionModleeModel):
       def __init__(self, img_size=(3, 32, 32)):
           super().__init__()
           # Store the input image size
           self.img_size = img_size
           
           # Calculate the total number of input features by multiplying dimensions
           self.input_dim = img_size[0] * img_size[1] * img_size[2]
           
           # Define a simple feed-forward neural network
           self.model = nn.Sequential(
               nn.Flatten(),  # Flatten the input image into a 1D vector
               nn.Linear(self.input_dim, 128),  # First linear layer with 128 output units
               nn.ReLU(),  # Apply ReLU activation function
               nn.Linear(128, 1)  # Output layer with a single unit (for regression)
           )
           
           # Define the loss function for training (Mean Squared Error for regression)
           self.loss_fn = nn.MSELoss()

       def forward(self, x):
           # Pass the input through the model to get the output
           x = self.model(x)
           return x
       
       def training_step(self, batch, batch_idx):
           # Unpack the batch into input features (x) and target labels (y)
           x, y = batch
           # Get the model predictions for the input
           logits = self.forward(x)
           # Calculate the training loss
           loss = self.loss_fn(logits, y)
           # Return the loss as a dictionary
           return {'loss': loss}
       
       def validation_step(self, val_batch):
           # Unpack the validation batch
           x, y_target = val_batch
           # Get the model predictions for the validation input
           y_pred = self.forward(x)
           # Calculate the validation loss
           val_loss = self.loss_fn(y_pred, y_target)
           # Return the validation loss as a dictionary
           return {'val_loss': val_loss}

       def configure_optimizers(self):
           # Configure the optimizer (Adam) with a learning rate of 1e-3
           return torch.optim.Adam(self.model.parameters(), lr=1e-3)

   # Create an instance of the model
   modlee_model = ModleeImageRegression(img_size=(3, 32, 32))

We start a training run with ``modlee.start_run()`` and configure the
``Trainer`` with the number of epochs. The model is then trained using
the ``trainer.fit()`` method with the specified dataloader.

.. code:: python

   # Start a training run with Modlee
   with modlee.start_run() as run:
       trainer = pl.Trainer(max_epochs=1)  # Set up the trainer
       trainer.fit(
           model=modlee_model,
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
   :target: https://www.kaggle.com/code/modlee/image-regression-example

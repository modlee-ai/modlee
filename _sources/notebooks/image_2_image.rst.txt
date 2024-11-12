|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Image2Image Example
===================

In this tutorial, we will walk through the process of building a deep
learning model using the Modlee package and PyTorch to denoise images
from the CIFAR-10 dataset.

The objective is to train a model that can learn to remove noise from
images, which is a common task in image processing.

|Open in Kaggle|

First, we will import the the necessary libraries and set up the
environment.

.. code:: python

   import torch
   import os
   import modlee
   import lightning.pytorch as pl
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms
   from torch import nn
   from utils import check_artifacts

Now, we will set up the ``modlee`` API key and initialize the ``modlee``
package. You can access your ``modlee`` API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__.

Replace ``replace-with-your-api-key`` with your API key.

.. code:: python

   modlee.init(api_key="replace-with-your-api-key")

To train our denoising model, we need to simulate noisy images. This is
done using a function called ``add_noise``, which takes an image and a
noise level as inputs.

We generate random noise and add it to the original image, ensuring that
the pixel values remain within the valid range of ``[0, 1]``.

.. code:: python

   def add_noise(img, noise_level=0.1):
       # Generate random noise with the specified noise level
       noise = torch.randn_like(img) * noise_level
       # Add noise to the original image and clamp the values to stay in range [0, 1]
       return torch.clamp(img + noise, 0., 1.)

We define a custom dataset class called ``NoisyImageDataset``, which
inherits from ``torch.utils.data.Dataset``. This class will help us
create a dataset that contains noisy images along with their clean
counterparts.

.. code:: python

   class NoisyImageDataset(torch.utils.data.Dataset):
       def __init__(self, dataset, noise_level=0.1, img_size=(1, 32, 32)):
           self.dataset = dataset  # Store the original dataset
           self.noise_level = noise_level  # Store the noise level
           self.img_size = img_size  # Store the target image size

       def __len__(self):
           return len(self.dataset)  # Return the size of the dataset

       def __getitem__(self, idx):
           img, _ = self.dataset[idx]  # Retrieve the image and ignore the label
           
           # Resize the image if necessary
           if img.size(0) != self.img_size[0]:
               if img.size(0) < self.img_size[0]:  
                   img = img.repeat(self.img_size[0] // img.size(0), 1, 1)  # Repeat channels to match size
               else:  
                   img = img[:self.img_size[0], :, :]  # Crop channels to match size

           # Resize the image to the target size
           img = transforms.Resize((self.img_size[1], self.img_size[2]))(img)  
           noisy_img = add_noise(img, self.noise_level)  # Create a noisy version of the image
           return noisy_img, img  # Return the noisy image and the clean image

Next, we create a model class called ``ModleeDenoisingModel``, which
extends ``modlee.model.ImageImageToImageModleeModel``. This class
defines the architecture of our neural network, which consists of
convolutional layers for feature extraction.

.. code:: python

   class ModleeDenoisingModel(modlee.model.ImageImageToImageModleeModel):
       def __init__(self, img_size=(1, 32, 32)):
           super().__init__()  # Initialize the parent class
           self.img_size = img_size  # Store the image size
           in_channels = img_size[0]  # Get the number of input channels
           # Define the model architecture
           self.model = nn.Sequential(
               nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),  # First convolutional layer
               nn.ReLU(),  # Activation function
               nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1)  # Second convolutional layer
           )
           self.loss_fn = nn.MSELoss()  # Define the loss function as Mean Squared Error

       def forward(self, x):
           return self.model(x)  # Define the forward pass

       def training_step(self, batch, batch_idx):
           x, y = batch  # Get the noisy images and their clean counterparts
           y_pred = self.forward(x)  # Get the model predictions
           loss = self.loss_fn(y_pred, y)  # Calculate the loss
           return {'loss': loss}  # Return the loss value

       def validation_step(self, val_batch, batch_idx):
           x, y_target = val_batch  # Get the validation batch
           y_pred = self.forward(x)  # Get the model predictions
           val_loss = self.loss_fn(y_pred, y_target)  # Calculate validation loss
           return {'val_loss': val_loss}  # Return the validation loss

       def configure_optimizers(self):
           return torch.optim.Adam(self.model.parameters(), lr=1e-3)  # Set up the optimizer

Now we need to create our datasets. We will use the ``CIFAR-10``
dataset, which consists of 60,000 32x32 color images in 10 different
classes.

To make our dataset suitable for training, we first define the
transformations to be applied to the images, which includes resizing and
converting them to tensors. We create both training and testing
datasets, applying our ``NoisyImageDataset`` class to introduce noise.

.. code:: python

   noise_level = 0.1  # Define the level of noise to add
   img_size = (3, 32, 32)  # Define the target image size (channels, height, width)

   # Define the transformations to be applied to the images
   transform = transforms.Compose([
       transforms.Resize((img_size[1], img_size[2])),  # Resize images to the target size
       transforms.ToTensor()  # Convert images to tensor format
   ])

   # Download and load the CIFAR-10 dataset
   train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
   test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

   # Create noisy datasets for training and testing
   train_noisy_dataset = NoisyImageDataset(train_dataset, noise_level=noise_level, img_size=img_size)
   test_noisy_dataset = NoisyImageDataset(test_dataset, noise_level=noise_level, img_size=img_size)

We then create ``DataLoader`` objects for both training and testing
datasets to enable batch processing during training.

.. code:: python

   # Create DataLoader for training and testing datasets
   train_dataloader = DataLoader(train_noisy_dataset, batch_size=2, shuffle=True)  # Shuffle training data
   test_dataloader = DataLoader(test_noisy_dataset, batch_size=2, shuffle=False)  # Do not shuffle test data

Now that we have our model and data prepared, we can begin training. We
instantiate the ``ModleeDenoisingModel``. We start a training run using
``modlee.start_run()``, which automatically logs the experiment details.

.. code:: python

   model = ModleeDenoisingModel(img_size=img_size)  # Instantiate the model 

   with modlee.start_run() as run:  # Start a training run
       trainer = pl.Trainer(max_epochs=1)  # Set up the trainer
       trainer.fit(  # Start training the model
           model=model,
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
   :target: https://www.kaggle.com/code/modlee/image2image

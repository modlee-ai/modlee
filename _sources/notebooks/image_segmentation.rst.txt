|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Image Segmentation
==================

In this tutorial, we will build an image segmentation model using the
``Pascal VOC 2012`` dataset, leveraging the Modlee package for
experimentation.

Steps Overview:

1. Setup and Initialization
2. Dataset Preparation
3. Model Definition
4. Model Training
5. Results and Artifacts Retrieval

|Open in Kaggle|

First, we will import the the necessary libraries and set up the
environment.

.. code:: python

   import torch
   import torchvision
   from torch.utils.data import DataLoader, Subset
   import lightning.pytorch as pl
   import modlee
   import os

Now, we will set up the ``modlee`` API key and initialize the ``modlee``
package. You can access your ``modlee`` API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__.

Replace ``replace-with-your-api-key`` with your API key.

.. code:: python

   modlee.init(api_key="replace-with-your-api-key")

Now, we will define transformations for the input images and
segmentation masks. Both will be resized to 256x256 pixels for
standardization.

.. code:: python

   # Define the transformations applied to the images and masks
   transform = torchvision.transforms.Compose([
       torchvision.transforms.Resize((256, 256)),  # Resize all images to 256x256
       torchvision.transforms.ToTensor()           # Convert images to PyTorch tensors
   ])

   target_transform = torchvision.transforms.Compose([
       torchvision.transforms.Resize((256, 256)),  # Resize the masks to match the image size
       torchvision.transforms.ToTensor()           # Convert masks to PyTorch tensors
   ])

Next, we will load the ``Pascal VOC 2012`` dataset using the
``VOCSegmentation`` class.

.. code:: python

   # Prepare the VOC 2012 dataset for segmentation tasks
   train_dataset = torchvision.datasets.VOCSegmentation(
       root='./data', year='2012', image_set='train', download=True, 
       transform=transform, #`transform` applies to input images
       target_transform=target_transform #`target_transform` applies to segmentation masks
   )

   val_dataset = torchvision.datasets.VOCSegmentation(
       root='./data', year='2012', image_set='val', download=True, 
       transform=transform,
       target_transform=target_transform
   )

To accelerate the training process, we will create smaller subsets of
the training and validation datasets. We will define a subset of 500
samples for training and 100 samples for validation.

.. code:: python

   # Use only a subset of the training and validation data to speed up training
   train_indices = torch.arange(500)  # Subset of 500 samples from training data
   val_indices = torch.arange(100)    # Subset of 100 samples from validation data

   # Create subsets of the datasets based on the indices we defined above
   train_subset = Subset(train_dataset, train_indices)
   val_subset = Subset(val_dataset, val_indices)

We will now create ``DataLoader`` instances for both the training and
validation subsets.

.. code:: python

   # Create DataLoader for both training and validation data
   train_dataloader = DataLoader(train_subset, batch_size=8, shuffle=True)
   val_dataloader = DataLoader(val_subset, batch_size=8, shuffle=False)

Next, we will create the image segmentation model within Modlee’s
framework, featuring an encoder for extracting relevant features and a
decoder for generating the segmentation mask.

.. code:: python

   # Define the image segmentation model
   class ImageSegmentation(modlee.model.ImageSegmentationModleeModel):
       def __init__(self, in_channels=3):
           super().__init__()
           # Encoder: A small convolutional neural network that processes the input image
           self.encoder = torch.nn.Sequential(
               torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
               torch.nn.ReLU(),  # Activation function to introduce non-linearity
               torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
               torch.nn.ReLU(),  # Another layer of convolution and activation
           )
           # Decoder: Upsampling to match the input size, producing a segmentation mask
           self.decoder = torch.nn.Sequential(
               torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsampling
               torch.nn.ReLU(),
               torch.nn.Conv2d(64, 21, kernel_size=1),  
           )
           # Loss function: Cross-entropy loss, commonly used for segmentation tasks
           self.loss_fn = torch.nn.CrossEntropyLoss()

       # Forward pass: Process the input through the encoder and decoder
       def forward(self, x):
           encoded = self.encoder(x)  # Apply encoder to input image
           decoded = self.decoder(encoded)  # Apply decoder to the encoded output
           output_size = x.shape[2:]  # Get the original image size to ensure the output matches it
           decoded = torch.nn.functional.interpolate(decoded, size=output_size, mode='bilinear', align_corners=False)
           return decoded  # Return the final segmentation mask

       # Training step: Called during each iteration of training
       def training_step(self, batch):
           x, y = batch  # Unpack the input images (x) and ground truth masks (y)
           logits = self.forward(x)  # Forward pass through the model
           loss = self.loss_fn(logits, y.squeeze(1).long())  # Compute the loss
           return loss  # Return the loss value for this batch

       # Validation step: Similar to training step but used for validation
       def validation_step(self, batch):
           x, y = batch  # Unpack the input images (x) and ground truth masks (y)
           logits = self.forward(x)  # Forward pass through the model
           loss = self.loss_fn(logits, y.squeeze(1).long())  # Compute the validation loss
           return loss  # Return the validation loss


       def configure_optimizers(self):
           return torch.optim.Adam(self.parameters(), lr=1e-3)


   model = ImageSegmentation(in_channels=3)  # Initialize the model

Now, we can train and evaluate our model using ``PyTorch Lightning`` for
one epoch.

.. code:: python

   # Train the model using Modlee and PyTorch Lightning
   with modlee.start_run() as run:
       # Set `max_epochs=1` to train for 1 epoch
       trainer = pl.Trainer(max_epochs=1)
       
       # Fit the model on the training and validation data
       trainer.fit(model=model, 
                   train_dataloaders=train_dataloader, 
                   val_dataloaders=val_dataloader)

After training, we will examine the artifacts saved by Modlee, such as
the model graph and various statistics. Modlee automatically preserves
your training assets, ensuring that valuable insights are available for
future reference and collaboration.

.. code:: python

   # Retrieve the path where Modlee saved the results of this run
   last_run_path = modlee.last_run_path()
   print(f"Run path: {last_run_path}")
   artifacts_path = os.path.join(last_run_path, 'artifacts')
   artifacts = sorted(os.listdir(artifacts_path))
   print(f"Saved artifacts: {artifacts}")

.. |Open in Kaggle| image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/code/modlee/modle-image-segmentation

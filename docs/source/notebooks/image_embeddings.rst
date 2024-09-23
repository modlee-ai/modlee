|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Image Embeddings With Tabular Classification Model
==================================================

In this example, we will walk through the process of building an image
classifier using embeddings from a pre-trained ResNet model combined
with a custom Multi-Layer Perceptron (MLP). We’ll train the MLP on
embeddings extracted from ResNet, which will handle feature extraction
from the CIFAR-10 dataset.

|Open in Colab|

First, we import the necessary libraries from ``PyTorch`` and
``Torchvision``.

.. code:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torchvision import models, transforms, datasets
   from torch.utils.data import DataLoader
   import os
   import cv2
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader, Subset, random_split
   from torchvision import datasets, models, transforms
   import modlee
   from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Now we will set our Modlee API key and initialize the Modlee package.
Make sure that you have a Modlee account and an API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__. Replace
``replace-with-your-api-key`` with your API key.

.. code:: python

   # Set the API key to an environment variable,
   # to simulate setting this in your shell profile
   os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
   modlee.init(api_key=os.environ['MODLEE_API_KEY'])

Next, we define a sequence of transformations to preprocess the images.
Images are resized to (224, 224) to match the input size required by the
pre-trained ``ResNet-50`` model.

.. code:: python

   transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4),
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

We load the ``CIFAR-10`` dataset and create a subset of 1,000 images for
faster experimentation. We then split it into training (80%) and
validation (20%) datasets using ``random_split``.

.. code:: python

   # Load the CIFAR-10 dataset with the specified transformations
   train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, 
                                   transform=transform)

   # Create a subset of the dataset for quicker experimentation
   subset_size = 1000
   indices = list(range(subset_size))
   subset_dataset = Subset(train_dataset, indices)

   # Split the subset into training and validation sets
   train_size = int(0.8 * len(subset_dataset))
   val_size = len(subset_dataset) - train_size
   train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

We define ``DataLoaders`` for both the training and validation datasets,
setting the batch size to 64.

.. code:: python

   # Create a DataLoader for the training dataset with shuffling enabled
   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

   # Create a DataLoader for the validation dataset without shuffling
   val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

We load a pre-trained ``ResNet-50`` model from ``torchvision.models``
and modify it to output image embeddings instead of predictions by
removing its fully connected (classification) layer.

.. code:: python

   # Load a pre-trained ResNet-50 model
   resnet = models.resnet50(pretrained=True)

   # Remove the final fully connected layer to get feature embeddings
   resnet = nn.Sequential(*list(resnet.children())[:-1]).to(device)

We define a custom Multi-Layer Perceptron (MLP) classifier using fully
connected layers, batch normalization, and dropout for regularization.

.. code:: python


   class MLP(modlee.model.TabularClassificationModleeModel):
       def __init__(self, input_size, num_classes):
           super().__init__()
           # Define the layers of the MLP model
           self.model = nn.Sequential(
               nn.Linear(input_size, 256),
               nn.BatchNorm1d(256),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(256, 128),
               nn.BatchNorm1d(128),
               nn.ReLU(),
               nn.Linear(128, num_classes)
           )
           self.loss_fn = nn.CrossEntropyLoss()

       def forward(self, x):
           return self.model(x)  # Forward pass through the MLP

       def training_step(self, batch):
           embeddings, labels = batch
           logits = self.forward(embeddings)  # Forward pass
           loss = self.loss_fn(logits, labels)  # Compute loss
           return loss

       def validation_step(self, batch):
           embeddings, labels = batch
           logits = self.forward(embeddings)  # Forward pass
           loss = self.loss_fn(logits, labels)  # Compute validation loss
           return loss

       def configure_optimizers(self):
           return torch.optim.Adam(self.parameters(), lr=1e-4)

We initialize our MLP model by passing the ``input_size`` of the
embeddings produced by ``ResNet-50`` and the ``num_classes`` for
classification. This model will map the 2048-dimensional embeddings to
the 10 class labels.

.. code:: python

   # Define the number of output classes for the classification task
   num_classes = 10

   # Initialize the MLP model with the specified input size and number of classes
   mlp_image = MLP(input_size=2048, num_classes=num_classes).to(device)

We pass the raw images through the pre-trained ``ResNet-50`` model,
which extracts high-level features from each image.

.. code:: python

   # Precompute embeddings using ResNet-50
   def precompute_embeddings(dataloader, model, device):
       model.eval()
       embeddings_list = []
       labels_list = []

       with torch.no_grad():
           for images, labels in dataloader:
               images = images.to(device)
               labels = labels.to(device)
               embeddings = model(images).squeeze()  # Extract features using ResNet
               embeddings_list.append(embeddings)
               labels_list.append(labels)

       return torch.cat(embeddings_list), torch.cat(labels_list)

.. code:: python

   # Precompute embeddings for training and validation datasets
   print("Precomputing embeddings for training and validation data")
   train_embeddings, train_labels = precompute_embeddings(train_loader, resnet, device)
   val_embeddings, val_labels = precompute_embeddings(val_loader, resnet, device)

   # Create TensorDataset for precomputed embeddings and labels
   train_embedding_dataset = TensorDataset(train_embeddings, train_labels)
   val_embedding_dataset = TensorDataset(val_embeddings, val_labels)

   # Create DataLoaders for the precomputed embeddings
   train_embedding_loader = DataLoader(train_embedding_dataset, batch_size=64, shuffle=True)
   val_embedding_loader = DataLoader(val_embedding_dataset, batch_size=64, shuffle=False)

We define the ``train_model`` function, which handles the training loop.

.. code:: python

   def train_model(model, dataloader, num_epochs=5):
       # Define the loss function and optimizer
       criterion = nn.CrossEntropyLoss()
       optimizer = optim.Adam(model.parameters(), lr=0.0001)

       for epoch in range(num_epochs):
           model.train()
           total_loss = 0
           correct = 0
           total = 0

           for embeddings, labels in dataloader:
               embeddings = embeddings.to(device)
               labels = labels.to(device)

               # Forward pass through the MLP model
               outputs = model(embeddings)
               loss = criterion(outputs, labels)

               # Perform backward pass and optimization
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()

               total_loss += loss.item()
               _, predicted = torch.max(outputs.data, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()

           # Print average loss and accuracy for the epoch
           print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, 
                   Accuracy: {100 * correct / total:.2f}%')

We define an ``evaluate_model`` function to evaluate the model’s
performance on the validation set.

.. code:: python

   def evaluate_model(model, dataloader):
       # Set the model to evaluation mode
       model.eval()
       with torch.no_grad():
           correct = 0
           total = 0

           for embeddings, labels in dataloader:
               embeddings = embeddings.to(device)
               labels = labels.to(device)

               # Forward pass through the MLP model
               outputs = model(embeddings)
               _, predicted = torch.max(outputs.data, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()

           # Print the accuracy of the model on the dataset
           print(f'Accuracy: {100 * correct / total:.2f}%')

After defining the model architecture and setting up the data loaders,
the final step involves training the model on the training dataset and
evaluating its performance on the validation set.

.. code:: python

   # Train and evaluate the model
   train_model(mlp_image, train_embedding_loader, num_epochs=5)
   evaluate_model(mlp_image, val_embedding_loader)

.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1ELBe98KV1uy0eHk1cL3rSiqEms0szAZe?usp=sharing&authuser=3

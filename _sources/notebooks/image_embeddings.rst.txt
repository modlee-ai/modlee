|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Image Embeddings With Tabular Classification Model
==================================================

In this example, we will walk through the process of building an image
classifier using embeddings from a pre-trained ``ResNet`` model combined
with a custom Multi-Layer Perceptron (MLP). We’ll train the MLP on
embeddings extracted from ``ResNet``, which will handle feature
extraction from the ``CIFAR-10`` dataset.

|Open in Kaggle|

First, we import the necessary libraries from ``PyTorch`` and
``Torchvision``.

.. code:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   import os
   import cv2
   from torchvision import datasets, models, transforms
   import modlee
   import lightning.pytorch as pl
   from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Now we will set our Modlee API key and initialize the Modlee package.
Make sure that you have a Modlee account and an API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__. Replace
``replace-with-your-api-key`` with your API key.

.. code:: python

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
   train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

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

   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
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
           return self.model(x)  

       def training_step(self, batch):
           embeddings, labels = batch
           logits = self.forward(embeddings)  
           loss = self.loss_fn(logits, labels)  
           return loss

       def validation_step(self, batch):
           embeddings, labels = batch
           logits = self.forward(embeddings)  
           loss = self.loss_fn(logits, labels)  
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
We also evaluate the model’s performance on the validation set.

.. code:: python

   def train_model(modlee_model, train_dataloader, val_dataloader, num_epochs=1):
       with modlee.start_run() as run:
           # Create a PyTorch Lightning trainer 
           trainer = pl.Trainer(max_epochs=num_epochs)

           # Train the model using the training and validation data loaders
           trainer.fit(
               model=modlee_model,
               train_dataloaders=train_dataloader,
               val_dataloaders=val_dataloader
           )

   # Train and evaluate the model
   train_model(mlp_image, train_embedding_loader, val_embedding_loader, num_epochs=5)

Finally, we can view the saved assets from training. With Modlee, your
training assets are automatically saved, preserving valuable insights
for future reference and collaboration.

.. code:: python

   last_run_path = modlee.last_run_path()
   print(f"Run path: {last_run_path}")
   artifacts_path = os.path.join(last_run_path, 'artifacts')
   artifacts = sorted(os.listdir(artifacts_path))
   print(f"Saved artifacts: {artifacts}")

.. |Open in Kaggle| image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/code/modlee/modlee-image-embeddings

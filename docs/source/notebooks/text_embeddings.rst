|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Text Embeddings With Tabular Classification Model
=================================================

This tutorial will guide you through a step-by-step breakdown of using a
Multilayer Perceptron (MLP) with embeddings from a pre-trained
``DistilBERT`` model to classify text sentiment from the IMDB movie
reviews dataset. We’ll cover everything from dataset preprocessing to
model evaluation, explaining each part in detail.

|Open in Colab|

In this section, we import the necessary libraries from ``PyTorch`` and
``Torchvision``.

.. code:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from transformers import DistilBertTokenizer, DistilBertModel
   from torch.utils.data import TensorDataset, DataLoader
   from datasets import load_dataset
   from sklearn.model_selection import train_test_split
   import modlee

.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1oXsl5RtlZFJ6nFu6brGfn639YdYF6PCu?usp=sharing

Now we will set our Modlee API key and initialize the Modlee package.
Make sure that you have a Modlee account and an API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__. Replace
``replace-with-your-api-key`` with your API key.

.. code:: python

   # Set the API key to an environment variable,
   # to simulate setting this in your shell profile
   import os

   os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
   modlee.init(api_key=os.environ['MODLEE_API_KEY'])

The MLP (Multilayer Perceptron) model is defined here as a neural
network with three fully connected linear layers. Each layer is followed
by a ``ReLU`` activation function.

.. code:: python

   class MLP(modlee.model.TabularClassificationModleeModel):
       def __init__(self, input_size, num_classes):
           super().__init__()
           self.model = nn.Sequential(
               nn.Linear(input_size, 256),  # First fully connected layer
               nn.ReLU(),                   # ReLU activation
               nn.Linear(256, 128),          # Second fully connected layer
               nn.ReLU(),                   # ReLU activation
               nn.Linear(128, num_classes)   # Output layer
           )
           self.loss_fn = nn.CrossEntropyLoss()

       def forward(self, x):
           # Pass input through the model defined in nn.Sequential
           return self.model(x)

       def training_step(self, batch, batch_idx):
           x, y_target = batch
           y_pred = self(x)
           loss = self.loss_fn(y_pred, y_target) # Calculate the loss
           return {"loss": loss}

       def validation_step(self, val_batch, batch_idx):
           x, y_target = val_batch
           y_pred = self(x)
           val_loss = self.loss_fn(y_pred, y_target)  # Calculate validation loss
           return {'val_loss': val_loss}

       def configure_optimizers(self):
           optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)  # Define the optimizer
           return optimizer

In this step, we load ``DistilBERT``, which is a more compact version of
``BERT``. The tokenizer is responsible for converting raw text into a
format that the ``DistilBERT`` model can understand.

.. code:: python

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Load the pre-trained DistilBERT tokenizer and model
   tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
   bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

We load the IMDB dataset using the Hugging Face datasets library.

.. code:: python

   dataset = load_dataset('imdb')

Since processing the entire dataset can be slow, we sample a subset of
1000 examples from the dataset to speed up computation.

.. code:: python

   def sample_subset(dataset, subset_size=1000):
       # Randomly shuffle dataset indices and select a subset
       sample_indices = torch.randperm(len(dataset))[:subset_size]
       # Select the sampled data based on the shuffled indices
       sampled_data = dataset.select(sample_indices.tolist())

       return sampled_data

``DistilBERT`` turns text into numerical embeddings that the model can
understand. We first preprocess the text by tokenizing and padding it.
Then, ``DistilBERT`` generates embeddings for each sentence.

.. code:: python

   def get_text_embeddings(texts, tokenizer, bert, device, max_length=128):

       # Tokenize the input texts, with padding and truncation to a fixed max length
       inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, 
                           max_length=max_length)
       input_ids = inputs['input_ids'].to(device)
       attention_mask = inputs['attention_mask'].to(device)

       # Get the embeddings from BERT without calculating gradients
       with torch.no_grad():
           # Average over the last hidden states to get sentence-level embeddings
           embeddings = bert(input_ids, 
                           attention_mask=attention_mask).last_hidden_state.mean(dim=1)
       return embeddings


   # Precompute embeddings for the entire dataset
   def precompute_embeddings(dataset, tokenizer, bert, device, max_length=128):
       texts = dataset['text']  # Extract texts from the dataset
       embeddings = get_text_embeddings(texts, tokenizer, bert, device, max_length)
       return embeddings

We use ``train_test_split`` to split the precomputed embeddings and
their corresponding labels into training and validation sets.

.. code:: python

   def split_data(embeddings, labels):
       # Split the embeddings and labels into training and validation sets 
       train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
           embeddings, labels, test_size=0.2, random_state=42
       )
       return train_embeddings, val_embeddings, train_labels, val_labels

The training and validation data are batched using the
``PyTorch DataLoader``, which ensures efficient processing during
training.

.. code:: python

   def create_dataloaders(train_embeddings, train_labels, val_embeddings, val_labels, batch_size):
       # Create TensorDataset objects for training and validation data
       train_dataset = TensorDataset(train_embeddings, train_labels)
       val_dataset = TensorDataset(val_embeddings, val_labels)

       # Create DataLoader objects to handle batching and shuffling of data
       train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
       val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 

       return train_loader, val_loader

The ``train_model`` function defines the training loop.

.. code:: python

   def train_model(model, train_loader, num_epochs=5):
       # Define the loss function and optimizer
       criterion = nn.CrossEntropyLoss()  # Loss function for classification
       optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer to update model weights

       # Iterate over epochs
       for epoch in range(num_epochs):
           model.train()  # Set the model to training mode
           running_loss = 0.0

           # Iterate over batches of data
           for embeddings, labels in train_loader:
               embeddings, labels = embeddings.to(device), labels.to(device)  

               # Forward pass: compute predictions and loss
               outputs = model(embeddings)
               loss = criterion(outputs, labels)

               # Backward pass and optimization
               optimizer.zero_grad()  # Clear previous gradients
               loss.backward()  # Compute gradients
               optimizer.step()  # Update model weights

               running_loss += loss.item() * embeddings.size(0)  # Accumulate loss

           # Print average loss for the epoch
           epoch_loss = running_loss / len(train_loader.dataset)
           print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

After training, the model is evaluated on the validation set.

.. code:: python

   def evaluate_model(model, val_loader):
       model.eval()  # Set the model to evaluation mode
       correct = 0
       total = 0

       with torch.no_grad():  # Disable gradient calculation for evaluation
           # Iterate over validation data
           for embeddings, labels in val_loader:
               embeddings, labels = embeddings.to(device), labels.to(device) 
               outputs = model(embeddings)  # Get model predictions
               _, predicted = torch.max(outputs.data, 1)  # Get the predicted class labels

               total += labels.size(0)  # Update total count
               correct += (predicted == labels).sum().item()  # Count correct predictions

       # Calculate and print accuracy
       accuracy = (correct / total) * 100
       print(f'Accuracy: {accuracy:.2f}%')

Finally, we run the script, which follows these steps: loading and
sampling the dataset, precomputing embeddings, training the MLP, and
evaluating the model.

.. code:: python

   if __name__ == "__main__":
       
       # Load and preprocess a subset of the IMDB dataset
       train_data = sample_subset(dataset['train'], subset_size=1000)  
       test_data = sample_subset(dataset['test'], subset_size=1000)  

       # Precompute BERT embeddings to speed up training
       print("Precomputing embeddings for training and testing data...")
       train_embeddings = precompute_embeddings(train_data, tokenizer, bert, device)  
       test_embeddings = precompute_embeddings(test_data, tokenizer, bert, device) 

       # Convert labels from lists to tensors
       train_labels = torch.tensor(train_data['label'], dtype=torch.long)
       test_labels = torch.tensor(test_data['label'], dtype=torch.long)  

       # Split the training data into training and validation sets
       train_embeddings, val_embeddings, train_labels, val_labels = split_data(train_embeddings, train_labels)  

       # Create DataLoader instances for batching data
       batch_size = 32  # Define batch size
       train_loader, val_loader = create_dataloaders(train_embeddings, train_labels, val_embeddings, val_labels, batch_size)  

       # Initialize and train the MLP model
       input_size = 768  # Output size of BERT embeddings
       num_classes = 2  
       mlp_text = MLP(input_size=input_size, num_classes=num_classes).to(device) 

       # Train the model
       print("Starting training...")
       train_model(mlp_text, train_loader, num_epochs=5) 

       # Evaluate the model's performance
       print("Evaluating model...")
       evaluate_model(mlp_text, val_loader) 

|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Text Embeddings With Tabular Classification Model
=================================================

This tutorial will guide you through a step-by-step breakdown of using a
Multilayer Perceptron (MLP) with embeddings from a pre-trained
``DistilBERT`` model to classify text sentiment from the IMDB movie
reviews dataset. We’ll cover everything from dataset preprocessing to
model evaluation, explaining each part in detail.

|Open in Kaggle|

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
   import lightning.pytorch as pl

.. |Open in Kaggle| image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/code/modlee/modlee-text-embeddings

Now we will set our Modlee API key and initialize the Modlee package.
Make sure that you have a Modlee account and an API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__. Replace
``replace-with-your-api-key`` with your API key.

.. code:: python

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
               nn.Linear(input_size, 256),  
               nn.ReLU(),                   
               nn.Linear(256, 128),        
               nn.ReLU(),                  
               nn.Linear(128, num_classes)   
           )
           self.loss_fn = nn.CrossEntropyLoss()

       def forward(self, x):
           return self.model(x)

       def training_step(self, batch, batch_idx):
           x, y_target = batch
           y_pred = self(x)
           loss = self.loss_fn(y_pred, y_target)
           return {"loss": loss}

       def validation_step(self, val_batch, batch_idx):
           x, y_target = val_batch
           y_pred = self(x)
           val_loss = self.loss_fn(y_pred, y_target)  
           return {'val_loss': val_loss}

       def configure_optimizers(self):
           optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)  
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
       inputs = tokenizer(texts, return_tensors="pt", padding=True, 
                           truncation=True, max_length=max_length)
       input_ids = inputs['input_ids'].to(device)
       attention_mask = inputs['attention_mask'].to(device)

       # Get the embeddings from BERT without calculating gradients
       with torch.no_grad():
           embeddings = bert(input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
       return embeddings


   # Precompute embeddings for the entire dataset
   def precompute_embeddings(dataset, tokenizer, bert, device, max_length=128):
       texts = dataset['text'] 
       embeddings = get_text_embeddings(texts, tokenizer, bert, device, max_length)
       return embeddings

We use ``train_test_split`` to split the precomputed embeddings and
their corresponding labels into training and validation sets.

.. code:: python

   def split_data(embeddings, labels):
       # Split the embeddings and labels into training and validation sets 
       train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
           embeddings, labels, test_size=0.2, random_state=42)
       
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

The ``train_model`` function defines the training loop. After training,
the model is evaluated on the validation set.

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

Finally, we run the script, which follows these steps: loading and
sampling the dataset, precomputing embeddings, training the MLP, and
evaluating the model.

.. code:: python

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
   input_size = 768  
   num_classes = 2   
   mlp_text = MLP(input_size=input_size, num_classes=num_classes).to(device) 

   print("Starting training...")
   train_model(mlp_text, train_loader, val_loader, num_epochs=5) 

We can view the saved training assests. With Modlee, your training
assets are automatically saved, preserving valuable insights for future
reference and collaboration.

.. code:: python

   last_run_path = modlee.last_run_path()
   print(f"Run path: {last_run_path}")
   artifacts_path = os.path.join(last_run_path, 'artifacts')
   artifacts = sorted(os.listdir(artifacts_path))
   print(f"Saved artifacts: {artifacts}")

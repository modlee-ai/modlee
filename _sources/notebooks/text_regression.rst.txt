Text Regression
===============

This tutorial will walk you through building an end-to-end text
regression pipeline using the ``Modlee`` package and
``PyTorch Lightning``.

We’ll use the ``Yelp Polarity`` dataset, which contains customer reviews
labeled with sentiment scores, to build a simple regression model that
predicts a continuous value based on the text.

|Open in Kaggle|

First, we will import the the necessary libraries and set up the
environment.

.. code:: python

   import os
   import torch
   import modlee
   import lightning.pytorch as pl
   from torch.utils.data import DataLoader, TensorDataset
   from sklearn.model_selection import train_test_split
   from transformers import AutoTokenizer
   from utils import get_device
   from datasets import load_dataset

Now, we will set up the ``modlee`` API key and initialize the ``modlee``
package. You can access your ``modlee`` API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__.

Replace ``replace-with-your-api-key`` with your API key.

.. code:: python

   modlee.init(api_key="replace-with-your-api-key")

Text data needs to be tokenized (converted into numerical format) before
it can be used by machine learning models. We use a pre-trained BERT
tokenizer for this.

.. code:: python

   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

We define a function to preprocess raw text data using the tokenizer.
Tokenization ensures that the input data has a uniform format and
length, making it suitable for training deep learning models.

.. code:: python

   def tokenize_texts(texts, tokenizer, max_length=20):
       encodings = tokenizer(
           texts,
           truncation=True,  # Shorten longer texts
           padding="max_length",  # Pad shorter texts to a fixed length
           max_length=max_length,
           return_tensors="pt",  # Return data as PyTorch tensors
           add_special_tokens=True,  # Include tokens like [CLS] and [SEP]
       )
       input_ids = encodings['input_ids'].to(torch.long)  # Token IDs
       attention_mask = encodings['attention_mask'].to(torch.long)  # Padding masks
       return input_ids, attention_mask

In this step, we load a text dataset using Hugging Face’s ``datasets``
library. We are using the **Yelp Polarity** dataset, which consists of
movie reviews labeled as positive or negative.

.. code:: python

   def load_real_data(dataset_name):
       # Load the dataset based on the provided name.
       # In this case, we are specifically loading the 'yelp_polarity' dataset.
       dataset = load_dataset("yelp_polarity", split='train[:80%]')  # Load the first 80% of the training data

       # Extract the 'text' column from the dataset, which contains the review texts.
       texts = dataset['text']
       
       # Extract the 'label' column, which contains the sentiment labels (positive or negative).
       targets = dataset['label']
       
       # Return the texts and their corresponding sentiment labels (targets).
       return texts, targets

We tokenize the dataset and split it into training and testing sets.
This step ensures that we have separate datasets for training and
evaluation.

.. code:: python

   # Load 'yelp_polarity' dataset
   texts, targets = load_real_data(dataset_name="yelp_polarity")

   # Use only the first 100 samples for simplicity
   texts = texts[:100]  
   targets = targets[:100]

   # Tokenize the text into input IDs and attention masks
   input_ids, attention_masks = tokenize_texts(texts, tokenizer)

   # Split the data into training and testing sets (80% train, 20% test)
   X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
       input_ids, attention_masks, targets, test_size=0.2, random_state=42
   )

We prepare PyTorch ``DataLoader`` objects to feed data into the model
during training.

.. code:: python

   # Create TensorDataset for training data
   train_dataset = TensorDataset(
       torch.tensor(X_train_ids, dtype=torch.long),
       torch.tensor(X_train_masks, dtype=torch.long),
       torch.tensor(y_train, dtype=torch.float)
   )

   # Create TensorDataset for testing data
   test_dataset = TensorDataset(
       torch.tensor(X_test_ids, dtype=torch.long),
       torch.tensor(X_test_masks, dtype=torch.long),
       torch.tensor(y_test, dtype=torch.float)
   )

   # Create DataLoader for training data
   train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

   # Create DataLoader for testing data
   test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

   # Add tokenizer to the training dataloader
   train_dataloader.initial_tokenizer = tokenizer

We create a custom text regression model by inheriting from Modlee’s
``TextRegressionModleeModel``.

.. code:: python

   class ModleeTextRegressionModel(modlee.model.TextRegressionModleeModel):
       def __init__(self, vocab_size, embed_dim=50, tokenizer=None):
           # Initialize the parent class to inherit Modlee functionality
           super().__init__()
           
           # Create an embedding layer to convert token IDs into dense vectors
           self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id if tokenizer else None)
           
           # Define the rest of the model architecture
           self.model = torch.nn.Sequential(
               self.embedding,  # Convert token IDs into embeddings
               torch.nn.Flatten(),  # Flatten the embedded vectors for linear layers
               torch.nn.Linear(embed_dim * 20, 128),  # Linear layer with 128 hidden units
               torch.nn.ReLU(),  # ReLU activation function for non-linearity
               torch.nn.Linear(128, 1)  # Output layer that produces a single regression value
           )
           
           # Define the loss function for regression (Mean Squared Error Loss)
           self.loss_fn = torch.nn.MSELoss()

       def forward(self, input_ids, attention_mask=None):
           # The forward pass takes input_ids (tokenized text) and attention_mask (if applicable)
           
           # If input_ids are provided as a list, concatenate them to form a single tensor
           if isinstance(input_ids, list):
               input_ids = torch.cat(input_ids, dim=0)
           
           # Pass the input_ids through the embedding layer
           embedded = self.embedding(input_ids)
           
           # Process the embedded vectors through the model's layers
           for layer in list(self.model.children())[1:]:  # Skip embedding layer (already applied)
               embedded = layer(embedded)  # Pass through each layer (Flatten, Linear, ReLU, etc.)
           
           return embedded  # Return the final prediction (single value)

       def training_step(self, batch, batch_idx):
           # This function is used during the training loop for a single batch
           input_ids, attention_mask, targets = batch  # Unpack the batch
           
           # Get the model predictions for the current batch
           preds = self.forward(input_ids, attention_mask)
           
           # Calculate the loss between the predictions and the true targets
           loss = self.loss_fn(preds.squeeze(), targets)  # Squeeze to remove any extra dimensions
           return loss  # Return the loss to be used by the optimizer

       def validation_step(self, batch, batch_idx):
           # This function is used during validation to calculate loss
           input_ids, attention_mask, targets = batch  # Unpack the batch
           
           # Get the model predictions for the current batch
           preds = self.forward(input_ids, attention_mask)
           
           # Calculate the validation loss between predictions and targets
           loss = self.loss_fn(preds.squeeze(), targets)  # Squeeze to remove any extra dimensions
           return loss  # Return the validation loss

       def configure_optimizers(self):
           # Configure the optimizer (Adam optimizer with learning rate of 1e-3)
           return torch.optim.Adam(self.parameters(), lr=1e-3)

We instantiate the model and use ``PyTorch Lightning’s Trainer`` class
to handle training.

.. code:: python

   # Initialize the model 
   modlee_model = ModleeTextRegressionModel(
       vocab_size=tokenizer.vocab_size, tokenizer=tokenizer
   ).to(device)

   # Train the model using Modlee and PyTorch Lightning's Trainer
   with modlee.start_run() as run:
       trainer = pl.Trainer(max_epochs=1) # Train for one epoch
       trainer.fit(
           model=modlee_model,
           train_dataloaders=train_dataloader,
           val_dataloaders=test_dataloader
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
   :target: https://www.kaggle.com/code/modlee/text-regression

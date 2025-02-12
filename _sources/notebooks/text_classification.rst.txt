Text Classification
===================

This tutorial will walk you through building an end-to-end text
classification pipeline using the ``Modlee`` package and
``PyTorch Lightning``.

We’ll use the ``Amazon Polarity`` dataset, which contains customer
reviews labeled as positive or negative, to build a simple binary
classification model.

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

Tokenization transforms raw text into input IDs and attention masks. We
define a helper function ``tokenize_texts`` to handle this process.

.. code:: python


   # Define a function to tokenize text data
   def tokenize_texts(texts, tokenizer, max_length=20):
       encodings = tokenizer(
           texts,
           truncation=True,  # Truncate sequences longer than max_length
           padding="max_length",  # Pad shorter sequences to max_length
           max_length=max_length,  # Maximum sequence length
           return_tensors="pt",  # Return PyTorch tensors
           add_special_tokens=True,  # Add special tokens like [CLS] and [SEP]
       )
       # Extract token IDs and attention masks as tensors
       input_ids = encodings['input_ids'].to(torch.long)
       attention_mask = encodings['attention_mask'].to(torch.long)
       return input_ids, attention_mask

The ``load_real_data`` function loads the Amazon Polarity dataset, which
contains customer reviews and their corresponding labels (positive or
negative). We extract the text data and labels, limiting the dataset to
100 samples for simplicity in this example.

.. code:: python

   def load_real_data(dataset_name="amazon_polarity"):
       dataset = load_dataset("amazon_polarity", split='train[:80%]')
       texts = dataset['content'] # Extract text data
       targets = dataset['label'] # Extract labels
       return texts, targets
       
   # Load and preprocess the dataset
   texts, targets = load_real_data(dataset_name="amazon_polarity")
   texts, targets = texts[:100], targets[:100]  # Use only the first 100 samples for simplicity

To evaluate the model, we split the data into training and testing
subsets. The ``train_test_split`` function ensures that 80% of the data
is used for training and 20% for testing.

.. code:: python

   # Tokenize the text data
   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   input_ids, attention_masks = tokenize_texts(texts, tokenizer)

   # Split the data into training and testing sets
   X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
       input_ids, attention_masks, targets, test_size=0.2, random_state=42
   )

DataLoaders enable efficient processing by dividing the dataset into
smaller batches for training. Here, we create separate DataLoaders for
the training and testing datasets.

.. code:: python

   # Create DataLoader objects for training and testing
   train_dataset = TensorDataset(
       torch.tensor(X_train_ids, dtype=torch.long),
       torch.tensor(X_train_masks, dtype=torch.long),
       torch.tensor(y_train, dtype=torch.long)
   )
   test_dataset = TensorDataset(
       torch.tensor(X_test_ids, dtype=torch.long),
       torch.tensor(X_test_masks, dtype=torch.long),
       torch.tensor(y_test, dtype=torch.long)
   )

   train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

We create a custom text classification model by inheriting from Modlee’s
``TextClassificationModleeModel``.

.. code:: python

   class ModleeTextClassificationModel(modlee.model.TextClassificationModleeModel):
       def __init__(self, vocab_size, embed_dim=50, num_classes=2, tokenizer=None):
           super().__init__()
           # Embedding layer to map words to dense vectors
           self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id if tokenizer else None)
           # Sequential model containing a Flatten layer and fully connected layers
           self.model = torch.nn.Sequential(
               self.embedding,
               torch.nn.Flatten(),  # Flatten embeddings for linear layers
               torch.nn.Linear(embed_dim * 20, 128),  # Fully connected layer
               torch.nn.ReLU(),  # ReLU activation
               torch.nn.Linear(128, num_classes)  # Output layer for classification
           )
           # Define the loss function (cross-entropy for classification)
           self.loss_fn = torch.nn.CrossEntropyLoss()
       
       def forward(self, input_ids, attention_mask=None):
           if isinstance(input_ids, list):
               input_ids = torch.cat(input_ids, dim=0)
           embedded = self.embedding(input_ids)  # Pass input through the embedding layer
           for layer in list(self.model.children())[1:]:  # Apply the rest of the layers
               embedded = layer(embedded)
           return embedded

       def training_step(self, batch, batch_idx):
           input_ids, attention_mask, labels = batch
           preds = self.forward(input_ids, attention_mask)  # Get predictions
           loss = self.loss_fn(preds, labels)  # Compute loss
           return loss

       def validation_step(self, batch, batch_idx):
           input_ids, attention_mask, labels = batch
           preds = self.forward(input_ids, attention_mask)
           loss = self.loss_fn(preds, labels)
           return loss

       def configure_optimizers(self):
           return torch.optim.Adam(self.parameters(), lr=1e-3)

We instantiate the model and use ``PyTorch Lightning’s Trainer`` class
to handle training.

.. code:: python

   # Initialize the model 
   modlee_model = ModleeTextClassificationModel(
       vocab_size=tokenizer.vocab_size, num_classes=2, tokenizer=tokenizer
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
   :target: https://www.kaggle.com/code/modlee/text-classification

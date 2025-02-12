Text2Text Example
=================

This tutorial provides a detailed walkthrough of building an end-to-end
sequence-to-sequence pipeline using the ``Modlee`` package and
``PyTorch Lightning``.

We’ll use the Romanian-English translation subset from the ``WMT16``
dataset to create a transformer-based model capable of translating
English sentences into Romanian.

|Open in Kaggle|

First, we will import the the necessary libraries and set up the
environment.

.. code:: python

   import os
   import modlee
   import lightning.pytorch as pl
   from utils import check_artifacts, get_device
   from torch.utils.data import DataLoader, TensorDataset
   from sklearn.model_selection import train_test_split
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from datasets import load_dataset
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

Now, we will set up the ``modlee`` API key and initialize the ``modlee``
package. You can access your ``modlee`` API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__.

Replace ``replace-with-your-api-key`` with your API key.

.. code:: python

   modlee.init(api_key="replace-with-your-api-key")

These constants determine the batch size (number of samples per batch)
and the maximum sequence length for tokenized inputs.

.. code:: python

   # Number of samples processed in each batch
   BATCH_SIZE = 16

   # Maximum number of tokens per sequence
   MAX_LENGTH = 50

Now, we will load and tokenize the dataset. We use
``datasets.load_dataset()`` to fetch a translation dataset. The
``AutoTokenizer`` tokenizes both English and Romanian texts, truncating
or padding them to max_length.

.. code:: python

   def load_dataset_and_tokenize(num_samples, dataset_name="wmt16", subset="ro-en", split="train[:80%]", max_length=50):
       """
       Load a dataset and tokenize it for text-to-text tasks.
       Args:
           num_samples (int): Number of samples to load.
           dataset_name (str): Dataset to use (default: "wmt16").
           subset (str): Specific subset of the dataset (e.g., Romanian-English).
           split (str): Portion of data to use (default: "train[:80%]").
           max_length (int): Maximum sequence length for tokenization.

       Returns:
           tuple: Encoded input IDs, target IDs, and tokenizer instance.
       """
       # Load the specified dataset from Hugging Face's Datasets library
       dataset = load_dataset(dataset_name, subset, split=split)

       # Select only the first `num_samples` samples for quick testing
       subset = dataset.select(range(num_samples))
       texts = [item['translation']['en'] for item in subset]  # Source texts (English)
       target_texts = [item['translation']['ro'] for item in subset]  # Target texts (Romanian)

       # Load a pre-trained tokenizer (e.g., T5 tokenizer)
       tokenizer = AutoTokenizer.from_pretrained("t5-small")

       # Tokenize the source texts
       encodings = tokenizer(
           texts,
           truncation=True,  # Truncate sequences to the specified max_length
           padding="max_length",  # Pad sequences to the max_length
           max_length=max_length,
           return_tensors="pt",  # Return tokenized data as PyTorch tensors
           add_special_tokens=True,  # Add necessary special tokens like <pad> or <eos>
       )

       # Tokenize the target texts similarly
       target_encodings = tokenizer(
           target_texts,
           truncation=True,
           padding="max_length",
           max_length=max_length,
           return_tensors="pt",
           add_special_tokens=True,
       )

       # Convert encodings to PyTorch tensors with the appropriate data type
       input_ids = encodings['input_ids'].to(torch.long)
       decoder_input_ids = target_encodings['input_ids'].to(torch.long)

       return input_ids, decoder_input_ids, tokenizer

This function splits the dataset into training and validation sets.
``train_test_split`` ensures the data is randomly divided, with 80% for
training and 20% for validation.

.. code:: python

   def create_dataloaders(input_ids, decoder_input_ids, test_size=0.2, batch_size=16):
       """
       Split data into training and validation sets and create PyTorch DataLoaders.
       Args:
           input_ids (Tensor): Input token IDs.
           decoder_input_ids (Tensor): Decoder token IDs.
           test_size (float): Proportion of data for validation (default: 20%).
           batch_size (int): Number of samples per batch (default: 16).

       Returns:
           tuple: DataLoaders for training and validation.
       """
       # Split the input and target data into training and validation sets
       X_train, X_val, y_train, y_val = train_test_split(
           input_ids, decoder_input_ids, test_size=test_size, random_state=42
       )

       # Wrap the training and validation data into PyTorch Datasets
       train_dataset = TensorDataset(X_train, X_train, y_train)
       val_dataset = TensorDataset(X_val, X_val, y_val)

       # Create DataLoaders to efficiently load batches of data
       train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
       val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

       return train_dataloader, val_dataloader

Next, we initialize a transformer-based model:

-  Embedding Layer: Converts token IDs into dense vectors of size
   ``d_model``.
-  Positional Encoding: Adds positional context to embeddings, as
   transformers are position-agnostic.
-  Transformer Module: Implements the encoder-decoder architecture.
-  Output Layer: A fully connected layer projects transformer outputs to
   the vocabulary space.

The ``_generate_positional_encoding`` function creates sinusoidal
encodings, which are added to token embeddings to capture sequence
order.

.. code:: python

   class TransformerSeq2SeqModel(modlee.model.TextTextToTextModleeModel):
       def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, max_length=50):
           # Initialize a Transformer-based sequence-to-sequence model.
           super(TransformerSeq2SeqModel, self).__init__()

           # Embedding layer converts token indices into dense vectors of size `d_model`.
           self.embedding = nn.Embedding(vocab_size, d_model)

           # Pre-compute positional encodings for input and target sequences to add positional context.
           self.positional_encoding = nn.Parameter(
               self._generate_positional_encoding(d_model, max_length), requires_grad=False
           )

           # Transformer module with customizable encoder and decoder configurations.
           self.transformer = nn.Transformer(
               d_model=d_model,              # Model dimensionality
               nhead=nhead,                  # Number of attention heads in multi-head attention
               num_encoder_layers=num_encoder_layers,  # Number of encoder layers
               num_decoder_layers=num_decoder_layers,  # Number of decoder layers
               dim_feedforward=2048,         # Size of the feedforward network
               dropout=0.1                   # Dropout rate for regularization
           )

           # Fully connected output layer maps Transformer outputs back to the vocabulary space.
           self.fc_out = nn.Linear(d_model, vocab_size)

           # Store model parameters for reference.
           self.max_length = max_length
           self.d_model = d_model

       def forward(self, input_ids, decoder_input_ids=None):
           # If no decoder input is provided, use the encoder input (e.g., for auto-regressive tasks).
           if decoder_input_ids is None:
               decoder_input_ids = input_ids

           # Handle case where inputs are provided as a tuple of encoder and decoder inputs.
           if isinstance(input_ids, list) and len(input_ids) == 2:
               (input_ids, decoder_input_ids) = input_ids

           # Add positional encoding to the embeddings for both source and target sequences.
           src = self.embedding(input_ids) * (self.d_model ** 0.5) + self.positional_encoding[:input_ids.size(1), :]
           tgt = self.embedding(decoder_input_ids) * (self.d_model ** 0.5) + self.positional_encoding[:decoder_input_ids.size(1), :]

           # Adjust dimensions to fit the Transformer module's expected input format (seq_len, batch, d_model).
           src = src.permute(1, 0, 2)
           tgt = tgt.permute(1, 0, 2)

           # Encode the source sequence to produce a memory representation.
           memory = self.transformer.encoder(src)

           # Decode the target sequence using the encoder's memory.
           output = self.transformer.decoder(tgt, memory)

           # Project the Transformer output to the vocabulary space and adjust dimensions back to (batch, seq_len, vocab_size).
           logits = self.fc_out(output.permute(1, 0, 2))
           return logits

       def training_step(self, batch, batch_idx):
           # Unpack the batch data into encoder inputs, decoder inputs, and labels.
           input_ids, decoder_input_ids, labels = batch

           # Forward pass through the model to generate logits.
           logits = self(input_ids, decoder_input_ids)

           # Compute cross-entropy loss between predictions and true labels.
           loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
           return loss

       def validation_step(self, batch, batch_idx):
           # Similar to the training step but used for validation.
           input_ids, decoder_input_ids, labels = batch
           logits = self(input_ids, decoder_input_ids)
           loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
           return loss

       def configure_optimizers(self):
           # Use the Adam optimizer with a learning rate of 5e-5 for training.
           return torch.optim.Adam(self.parameters(), lr=5e-5)

       @staticmethod
       def _generate_positional_encoding(d_model, max_length):
           # Generate sinusoidal positional encodings based on sequence position and model dimensionality.
           pos = torch.arange(0, max_length).unsqueeze(1)
           i = torch.arange(0, d_model, 2)
           angle_rates = 1 / torch.pow(10000, (i.float() / d_model))

           # Initialize positional encodings and calculate sine and cosine functions for even and odd indices.
           pos_enc = torch.zeros(max_length, d_model)
           pos_enc[:, 0::2] = torch.sin(pos * angle_rates)
           pos_enc[:, 1::2] = torch.cos(pos * angle_rates)

           return pos_enc

We instantiate the model and use ``PyTorch Lightning’s Trainer`` class
to handle training. The Trainer manages training loops, validation, and
logging.

.. code:: python

   # Load data and tokenize it
   input_ids, decoder_input_ids, tokenizer = load_dataset_and_tokenize(num_samples=100)

   # Create data loaders for training and validation
   train_dataloader, val_dataloader = create_dataloaders(input_ids, decoder_input_ids)

   # Initialize the transformer model with the tokenizer's vocabulary size
   model = TransformerSeq2SeqModel(vocab_size=tokenizer.vocab_size)

   # Use PyTorch Lightning's Trainer to handle training and validation
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
   :target: https://www.kaggle.com/code/modlee/text2text

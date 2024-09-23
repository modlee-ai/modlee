|image0|

.. |image0| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Audio Embeddings With Tabular Classification Model
==================================================

In this example, we will build an audio classification model using
``PyTorch`` and ``Wav2Vec2``, a pretrained model for processing audio
data. This guide will walk you through each step of the process,
including setting up the environment, loading and preprocessing data,
defining and training a model, and evaluating its performance.

|Open in Colab|

First, we will import the the necessary libraries and set up the
environment.

.. code:: python

   import torchaudio
   from sklearn.preprocessing import LabelEncoder
   from torch.utils.data import TensorDataset, DataLoader
   from transformers import Wav2Vec2Model
   from torch.utils.data import DataLoader, TensorDataset
   import torch
   import os
   import modlee
   import lightning.pytorch as pl
   from sklearn.model_selection import train_test_split
   torchaudio.set_audio_backend("sox_io")

Now we will set our Modlee API key and initialize the Modlee package.
Make sure that you have a Modlee account and an API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__. Replace
``replace-with-your-api-key`` with your API key.

.. code:: python

   # Set the API key to an environment variable,
   # to simulate setting this in your shell profile
   os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
   modlee.init(api_key=os.environ['MODLEE_API_KEY'])

Now, we will prepare our data. For this example, we will manually
download the ``Human Words`` dataset from Kaggle and upload it to the
environment.

Visit the `Human Words Audio dataset
page <https://www.kaggle.com/datasets/warcoder/cats-vs-dogs-vs-birds-audio-classification?resource=download>`__
on Kaggle and click the **Download** button to save the ``Animals``
directory to your local machine.

Copy the path to that donwloaded file, which will be used later. This
snippet loads the ``Wav2Vec2 model``. We’ll use it to convert audio into
embeddings.

This snippet loads the ``Wav2Vec2`` model. ``Wav2Vec2`` is a model
designed for speech processing. We’ll use it to convert audio into
embeddings.

.. code:: python

   # Set device to GPU if available, otherwise use CPU.
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Load the pre-trained Wav2Vec2 model and move it to the specified device.
   wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

This function converts raw audio waveforms into embeddings using the
``Wav2Vec2`` model.

.. code:: python

   def get_wav2vec_embeddings(waveforms):
       with torch.no_grad():  # Turn off gradients to save memory during inference
           # Convert waveforms to a tensor and move it to the chosen device
           inputs = torch.tensor(waveforms).to(device)
           # Get embeddings from the Wav2Vec2 model
           embeddings = wav2vec(inputs).last_hidden_state.mean(dim=1)
       return embeddings

The ``AudioDataset`` class handles loading and preprocessing of audio
files.

.. code:: python

   class AudioDataset(TensorDataset):
       def __init__(self, audio_paths, labels, target_length=16000):
           self.audio_paths = audio_paths  # List of paths to audio files
           self.labels = labels  # List of labels corresponding to audio files
           self.target_length = target_length  # Desired length for audio clips

       def __len__(self):
           return len(self.audio_paths)  # Number of items in the dataset

       def __getitem__(self, idx):
           audio_path = self.audio_paths[idx]  # Get the path of the audio file
           label = self.labels[idx]  # Get the label for the audio file
           waveform, sample_rate = torchaudio.load(audio_path, normalize=True)  
           waveform = waveform.mean(dim=0)  # Convert to mono by averaging channels

           # Pad or truncate the waveform to the target length
           if waveform.size(0) < self.target_length:
               waveform = torch.cat([waveform, torch.zeros(self.target_length - waveform.size(0))])
           else:
               waveform = waveform[:self.target_length]

           return waveform, label  

This function loads audio files and their corresponding labels from a
directory structure.

.. code:: python

   def load_dataset(data_dir):
       audio_paths = []  # List to store paths to audio files
       labels = []  # List to store labels corresponding to each audio file

       # Loop through each subdirectory in the data directory
       for label_dir in os.listdir(data_dir):
           label_dir_path = os.path.join(data_dir, label_dir)
           if os.path.isdir(label_dir_path):  # Check if it's a directory
               # Loop through each file in the directory
               for file_name in os.listdir(label_dir_path):
                   if file_name.endswith('.wav'):  # Check if the file is a .wav file
                       audio_paths.append(os.path.join(label_dir_path, file_name))  
                       labels.append(label_dir)  # Add label (directory name) to list

       return audio_paths, labels  # Return lists of file paths and labels

We define a simple Multi-Layer Perceptron (MLP) model for
classification. This model takes the embeddings from ``Wav2Vec2`` as
input.

.. code:: python

   class MLP(modlee.model.TabularClassificationModleeModel):
       def __init__(self, input_size, num_classes):
           super().__init__()
           # Define the model using nn.Sequential for simplicity
           self.model = torch.nn.Sequential(
               torch.nn.Linear(input_size, 256),  # First fully connected layer
               torch.nn.ReLU(),                   # ReLU activation
               torch.nn.Linear(256, 128),          # Second fully connected layer
               torch.nn.ReLU(),                   # ReLU activation
               torch.nn.Linear(128, num_classes)   # Output layer
           )
           self.loss_fn = torch.nn.CrossEntropyLoss()

       def forward(self, x):
           # Forward pass through the model
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

``Wav2Vec2`` transforms raw audio data into numerical embeddings that a
model can interpret. We preprocess the audio by normalizing and padding
it to a fixed length. Then, ``Wav2Vec2`` generates embeddings for each
audio clip.

.. code:: python

   def precompute_embeddings(dataloader):
       embeddings_list = []
       labels_list = []
       for inputs, labels in dataloader:
           inputs = inputs.to(device)
           embeddings = get_wav2vec_embeddings(inputs)  # Precompute embeddings
           embeddings_list.append(embeddings.cpu())
           labels_list.append(labels)
       embeddings_list = torch.cat(embeddings_list, dim=0)  # Stack all embeddings
       labels_list = torch.cat(labels_list, dim=0)  # Stack all labels
       return embeddings_list, labels_list

We create functions to train and evaluate our model.

.. code:: python

   def train_model(model, dataloader, num_epochs=1):
       # Define the loss function and optimizer
       criterion = torch.nn.CrossEntropyLoss()
       optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
       model.train()  # Set the model to training mode

       for epoch in range(num_epochs):
           running_loss = 0.0
           for embeddings, labels in dataloader:
               embeddings = embeddings.to(device)
               labels = labels.to(device)

               optimizer.zero_grad()  # Clear previous gradients
               outputs = model(embeddings)  # Get model predictions
               loss = criterion(outputs, labels)  # Compute the loss
               loss.backward()  # Backpropagate the loss
               optimizer.step()  # Update model weights

               running_loss += loss.item()  # Accumulate loss

           # Print average loss for the epoch
           print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

.. code:: python

   def evaluate_model(model, dataloader):
       model.eval()  # Set the model to evaluation mode
       correct = 0
       total = 0

       with torch.no_grad():  # Disable gradient calculation
           for embeddings, labels in dataloader:
               embeddings = embeddings.to(device)
               labels = labels.to(device)

               outputs = model(embeddings)  # Get model predictions
               _, predicted = torch.max(outputs, 1)  # Get predicted class labels
               total += labels.size(0)  # Update total count
               correct += (predicted == labels).sum().item()  # Count correct predictions

       accuracy = correct / total  # Compute accuracy
       print(f'Accuracy: {accuracy * 100:.2f}%')  # Print accuracy percentage

Finally, we load the dataset, preprocess it, and train the model.

Add your path to the dataset in ``data_dir``.

.. code:: python

   if __name__ == "__main__":
       # Path to dataset
       data_dir = 'path-to-dataset'  # Use the dataset containing 'cats', 'dogs', 'birds'

       # Load dataset
       audio_paths, labels = load_dataset(data_dir)

       # Encode labels
       label_encoder = LabelEncoder()
       labels = label_encoder.fit_transform(labels)

       # Split dataset into training and validation sets
       train_paths, val_paths, train_labels, val_labels = train_test_split(audio_paths, labels, 
                                                                           test_size=0.2, random_state=42)

       # Create datasets and dataloaders
       target_length = 16000  # Define the length for padding/truncation
       train_dataset = AudioDataset(train_paths, train_labels, target_length=target_length)
       val_dataset = AudioDataset(val_paths, val_labels, target_length=target_length)
       train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
       val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

       # Precompute embeddings
       print("Precomputing embeddings for training and validation data...")
       train_embeddings, train_labels = precompute_embeddings(train_dataloader)
       val_embeddings, val_labels = precompute_embeddings(val_dataloader)

       # Create TensorDataset for precomputed embeddings and labels
       train_embedding_dataset = TensorDataset(train_embeddings, train_labels)
       val_embedding_dataset = TensorDataset(val_embeddings, val_labels)

       # Create DataLoaders for the precomputed embeddings
       train_embedding_loader = DataLoader(train_embedding_dataset, batch_size=4, shuffle=True)
       val_embedding_loader = DataLoader(val_embedding_dataset, batch_size=4, shuffle=False)

       # Define number of classes
       num_classes = len(label_encoder.classes_)
       mlp_audio = MLP(input_size=768, num_classes=num_classes).to(device)

       # Train and evaluate the model
       train_model(mlp_audio, train_embedding_loader)
       evaluate_model(mlp_audio, val_embedding_loader)

.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1tjrD_tUB7tbQuR6kJ_mPM-_kCbVY-Q71?usp=sharing#scrollTo=Ys9Rj0sVqrl8

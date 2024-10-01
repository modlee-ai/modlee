|image1|

.. |image1| image:: https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg

Audio Embeddings With Tabular Classification Model
==================================================

In this example, we will build an audio classification model using
``PyTorch`` and ``Wav2Vec2``, a pretrained model for processing audio
data. This guide will walk you through each step of the process,
including setting up the environment, loading and preprocessing data,
defining and training a model, and evaluating its performance.

|Open in Kaggle|

First, we will import the the necessary libraries and set up the
environment.

.. code:: python

   import torchaudio
   from sklearn.preprocessing import LabelEncoder
   from torch.utils.data import TensorDataset, DataLoader
   from transformers import Wav2Vec2Model
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

   os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
   modlee.init(api_key=os.environ['MODLEE_API_KEY'])

Now, we will prepare our data. For this example, we will manually
download the ``Human Words Audio`` dataset from Kaggle and upload it to
the environment.

Visit the `Human Words Audio dataset
page <https://www.kaggle.com/datasets/warcoder/cats-vs-dogs-vs-birds-audio-classification?resource=download>`__
on Kaggle and click the **Download** button to save the ``Animals``
directory to your local machine.

Copy the path to that donwloaded file, which will be used later. This
snippet loads the ``Wav2Vec2`` model. We’ll use it to convert audio into
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
       with torch.no_grad():  
           inputs = torch.tensor(waveforms).to(device)
           embeddings = wav2vec(inputs).last_hidden_state.mean(dim=1)
       return embeddings

The ``AudioDataset`` class handles loading and preprocessing of audio
files.

.. code:: python

   class AudioDataset(TensorDataset):
       def __init__(self, audio_paths, labels, target_length=16000):
           self.audio_paths = audio_paths
           self.labels = labels 
           self.target_length = target_length  

       def __len__(self):
           return len(self.audio_paths) 

       def __getitem__(self, idx):
           audio_path = self.audio_paths[idx] 
           label = self.labels[idx]  
           waveform, sample_rate = torchaudio.load(audio_path, normalize=True) 
           waveform = waveform.mean(dim=0) 

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
       audio_paths = []  
       labels = []  

       # Loop through each subdirectory in the data directory
       for label_dir in os.listdir(data_dir):
           label_dir_path = os.path.join(data_dir, label_dir)
           if os.path.isdir(label_dir_path): 
               # Loop through each file in the directory
               for file_name in os.listdir(label_dir_path):
                   if file_name.endswith('.wav'):  
                       audio_paths.append(os.path.join(label_dir_path, file_name))  
                       labels.append(label_dir)  

       return audio_paths, labels 

We define a simple Multi-Layer Perceptron (MLP) model for
classification. This model takes the embeddings from ``Wav2Vec2`` as
input.

.. code:: python

   class MLP(modlee.model.TabularClassificationModleeModel):
       def __init__(self, input_size, num_classes):
           super().__init__()
           self.model = torch.nn.Sequential(
               torch.nn.Linear(input_size, 256),  
               torch.nn.ReLU(),                
               torch.nn.Linear(256, 128),          
               torch.nn.ReLU(),                   
               torch.nn.Linear(128, num_classes)   
           )
           self.loss_fn = torch.nn.CrossEntropyLoss()

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
           embeddings = get_wav2vec_embeddings(inputs)
           embeddings_list.append(embeddings.cpu())
           labels_list.append(labels)
       embeddings_list = torch.cat(embeddings_list, dim=0) 
       labels_list = torch.cat(labels_list, dim=0)  
       return embeddings_list, labels_list

We create a function to train and evaluate our model.

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

Finally, we load the dataset, preprocess it, and train the model.

Add your path to the dataset in ``data_dir``.

.. code:: python

   # Path to dataset
   data_dir = 'path-to-dataset'  

   # Load dataset
   audio_paths, labels = load_dataset(data_dir)

   # Encode labels
   label_encoder = LabelEncoder()
   labels = label_encoder.fit_transform(labels)

   # Split dataset into training and validation sets
   train_paths, val_paths, train_labels, val_labels = train_test_split(audio_paths, labels, 
                                                               test_size=0.2, random_state=42)

   # Create datasets and dataloaders
   target_length = 16000  
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
   train_model(mlp_audio, train_embedding_loader,val_embedding_loader)

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
   :target: https://www.kaggle.com/code/modlee/modlee-audio-embeddings

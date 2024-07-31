.. image:: https://github.com/mansiagr4/gifs/raw/main/logo%20only%20(2).svg
   :width: 50px
   :height: 50px

Model definition guidelines
===========================

Here we show pseudo code to illustrate how to define a ModleeModel that
is compatible with **Modlee Auto Experiment Documentation**. Following
these guidelines will ensure your model architecture is preserved, and
can be shared with your collaborators in a way that it can be reloaded,
*without having to share your model weights*.

TLDR
----

-  Define all custom objects as ``nn.module`` PyTorch classes within the
   same file as your ModleeModel.
-  Define all parameters used by PyTorch classes as hard coded variables
   within each class.

Define example custom ModleeModel
---------------------------------

.. code:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   import modlee

   # Define custom activation function
   class CustomActivation(nn.Module):
       def __init__(self):
           super(CustomActivation, self).__init__()

       def forward(self, x):
           return torch.log(torch.abs(x) + 1)

   # Define the CNN model with custom activation
   class CNNModel(nn.Module):
       def __init__(self):
           super(CNNModel, self).__init__()
           self.kernel_size = 3 # --- Hard coded parameter to define network paramters
           self.conv1 = nn.Conv2d(1, 32, kernel_size=self.kernel_size, padding=1)
           self.conv2 = nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=1)
           self.conv3 = nn.Conv2d(64, 128, kernel_size=self.kernel_size, padding=1)
           self.pool = nn.MaxPool2d(2, 2)
           self.activation = CustomActivation()  # Custom activation
           self.fc1 = nn.Linear(128 * 3 * 3, 512)
           self.fc2 = nn.Linear(512, 10)

       def forward(self, x):
           x = self.activation(self.conv1(x))
           x = self.pool(x)
           x = self.activation(self.conv2(x))
           x = self.pool(x)
           x = self.activation(self.conv3(x))
           x = self.pool(x)
           x = torch.flatten(x, 1)
           x = self.activation(self.fc1(x))
           x = self.fc2(x)
           return x

   # Define the ModleeModel
   class CNNModleeModel(modlee.model.ModleeModel):
       def __init__(self):
           super(CNNModleeModel, self).__init__()
           self.cnn = CNNModel()

       def forward(self, x):
           return self.cnn(x)

       def training_step(self, batch, batch_idx):
           x, y = batch
           y_hat = self(x)
           loss = nn.functional.cross_entropy(y_hat, y)
           self.log('train_loss', loss)
           return loss

       def configure_optimizers(self):
           return optim.Adam(self.parameters(), lr=0.001)

   # Initialize the ModleeModel
   model = CNNModleeModel()

MODLEE_GUIDELINE
----------------

Define all custom objects, layers, activations, etc .. as ``nn.module``
PyTorch classes within the same file as your ModleeModel. Doing so
ensures we can retrieve the necessary information to preserve the model
architecture.

Define all parameters, batch size, number of layers, learning rate, etc
…, used by PyTorch classes as hard coded variables within each class.
Avoid using YAMLs or other config files as inputs to your Pytorch
classes, because this makes it hard for us to retrieve parameter values
needed to preserve the model architecture.

Modlee preserves the architecture of your models so your experiments can
be used in aggregate analysis alongside your collaborators data, to
improve Modlee’s model recommendation technology **without sharing your
model weights**. This helps Modlee users protect parts of their IP,
trained models, while allowing them the freedom to share aspects of
their experiments to collaborate more effectively.

Modality & task compatibility
-----------------------------

We’re working on making modlee compatible with any data modality and
machine learning task which drove us to create the above stated
MODLEE_GUIDELINES.

Check out our `Github Repo <https://github.com/modlee-ai/modlee>`__ to
see which have been tested for auto documentation to date, and if you
don’t see one you need, test it out yourself and contribute!

Reach out on our `Discord <https://discord.com/invite/m8YDbWDvrF>`__ to
let us know what modality & tasks you want to use next, or give us
feedback on these guidelines.

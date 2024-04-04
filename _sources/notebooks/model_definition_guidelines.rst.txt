Model definition guidelines
===========================

Here we show pseudo code to illustrate building defining a ModleeModel
that is compatible with **Modlee Auto Experiment Documentation**.
Following these guidelines will ensure your model architecture is
preserved, and can be shared with your collaborates in a way that can be
reloaded, *without having to share your model weights*.

TLDR
----

-  Define all custom layers, activations, etc .. as ``nn.module``
   PyTorch classes within the same file as your ModleeModel. Doing so
   ensures we can retrieve the necessary information to preserve the
   model architecture.

-  Be careful of using YAMLs or other config files

Define example custom ModleeModel
---------------------------------

.. code:: ipython3

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
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
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


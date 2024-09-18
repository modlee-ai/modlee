api_key = "YAP0xPSv38ASnrm2UjehtWY09f7or6e2"

import os,zipfile,shutil,requests

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.jit as jit
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import Trainer, LightningModule
import lightning.pytorch as pl
import modlee
from torch.utils.data import DataLoader, TensorDataset
from modlee.model import ImageClassificationModleeModel

#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('cpu')
# print(f"Using device: {device}")

modlee.init(api_key=api_key)

def generate_dummy_data(num_samples=100, num_classes=2, img_size=(3, 32, 32)):
    # Ensure data is on the correct device and in float32 format
    X = torch.randn(num_samples, *img_size)  # Ensure float32
    y = torch.randint(0, num_classes, (num_samples,))  # Ensure proper dtype for classification
    return X, y


class cnnModel(ImageClassificationModleeModel):
    def __init__(self, num_classes=2, img_size=(3, 32, 32)):
        super().__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Define the CNN model
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=img_size[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the flattened dimension after the convolutional layers
        self.flattened_dim = 128 * (img_size[1] // 8) * (img_size[2] // 8)
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 128),  # Adjust the input dimension based on the output size of conv layers
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        #print(f"Original input shape: {x.shape}, total size: {x.numel()}")
        
        # If the shape is `[2, 32, 3, 32, 32]`, reshape it correctly
        if x.dim() == 5 and x.size(1) == 32:
            # Reshape to collapse the second dimension (32) into the batch dimension
            x = x.view(-1, 3, 32, 32)  # Adjust to correct shape: [batch_size, channels, height, width]
            print(f"Reshaped input shape: {x.shape}")

        # Forward pass through the model
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        
        #print(f"Shape after forward pass: {x.shape}")
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        print(f"Batch x shape: {x.shape}, Batch y shape: {y.shape}")
        
        # Forward pass
        logits = self.forward(x)
        
        # Calculate loss
        loss = self.loss_fn(logits, y)
        print(f"Loss: {loss.item()}")
        
        return loss
    
    def configure_optimizers(self):
        # Make sure to refer to self.model for the optimizer
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ModleeImageClassification(modlee.model.ImageClassificationModleeModel):
    def __init__(self, num_classes=2, img_size=(3, 32, 32)):
        super().__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.img_size = img_size
        input_dim = img_size[0] * img_size[1] * img_size[2]  # 3 * 32 * 32 = 3072
        
        # Define the model as a sequential set of layers
        self.model = nn.Sequential(
            nn.Flatten(),  # Flatten [batch_size, 3, 32, 32] -> [batch_size, 3072]
            nn.Linear(input_dim, 128),  # Linear layer with input 3072, output 128
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output layer with input 128, output num_classes
        )
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        print(f"Original input shape: {x.shape}, total size: {x.numel()}")
        print(f"Type of data being passed: {x.dtype}")
        
        # If the shape is `[2, 32, 3, 32, 32]`, reshape it correctly
        if x.dim() == 5 and x.size(1) == 32:
            # Reshape to collapse the second dimension (32) into the batch dimension
            x = x.view(-1, 3, 32, 32)  # Adjust to correct shape: [batch_size, channels, height, width]
            print(f"Reshaped input shape: {x.shape}")

        # Forward pass through the model
        x = self.model(x)
        
        print(f"Shape after forward pass: {x.shape}")
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        print(f"Batch x shape: {x.shape}, Batch y shape: {y.shape}")
        
        # Forward pass
        logits = self.forward(x)
        
        # Calculate loss
        loss = self.loss_fn(logits, y)
        print(f"Loss: {loss.item()}")
        
        return loss
    
    def configure_optimizers(self):
        # Make sure to refer to self.model for the optimizer
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)



# Generate dummy data and move it to the correct device
X_train, y_train = generate_dummy_data(num_samples=100, img_size=(3, 32, 32))
X_test, y_test = generate_dummy_data(num_samples=20, img_size=(3, 32, 32))
print(y_train.shape)
print(X_train.shape)
# Create TensorDataset and DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Initialize the model and move it to the MPS device
#modlee_model = LightningImageClassification(num_classes=2).to(device)
modlee_model = ModleeImageClassification(num_classes=2)
# modlee_model = cnnModel(num_classes=2).to(device)
# Ensure the model parameters are in float32
# modlee_model = modlee_model.to(torch.float32)
# Set up the trace function
# import sys
# def trace_calls(frame, event, arg):
#     if event != 'call':
#         return
#     co = frame.f_code
#     func_name = co.co_name
#     func_line_no = frame.f_lineno
#     func_filename = co.co_filename
#     print(f"Call to {func_name} on line {func_line_no} of {func_filename}")
#     return trace_calls

# Enable tracing

# Train the model using PyTorch Lightning's Trainer
with modlee.start_run() as run:
    
    trainer = pl.Trainer(max_epochs=1)
    # print("Modlee set on device {} during traing loop".format(modlee_model.device))
    #print("Input set on device {} during traing loop".format(train_dataset.device))
    trainer.fit(
        model=modlee_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader
    )

    #sys.settrace(trace_calls)


#submit(api_key,exercise_id,trainer.model,example_batch_images,modlee)
#print('As a reminder, your exercises model_size_restriction_MB is ',model_size_restriction_MB)
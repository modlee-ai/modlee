api_key = "YAP0xPSv38ASnrm2UjehtWY09f7or6e2"

import os,zipfile,shutil,requests

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.jit as jit
import torch.nn.functional as F
import numpy as np

import lightning.pytorch as pl
import modlee
from modlee.recommender import from_modality_task as trainer_for_modality_task

exercise_id = 'D_B_L__42046588550'
exercise_modality = 'image'
exercise_task = 'classification'
model_size_restriction_MB = '15'


root_url = 'https://evalserver.modlee.ai:5000'
url = f"{root_url}/get-interview-utils"  # Change the port if your Flask app is running on a different one
response = requests.get(url, params={'api_key': api_key,'exercise_id':exercise_id})

# Check if the request was successful
if response.status_code == 200:
    with open('interview_utils.py', 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("File downloaded and saved as interview_utils.py")
else:
    print("Failed to download file:", response.status_code)
from interview_utils import *
from interview_utils import ModleeImageClassifier,setup,submit

unzip_train_dataloader,unzip_val_dataloader,example_batch_images = setup(api_key,exercise_id)

modlee.init(api_key=api_key)


class ExampleCNN(nn.Module):
    def __init__(self, num_classes):
        super(ExampleCNN, self).__init__()
        # Single convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        # Single fully connected (dense) layer
        self.fc1 = nn.Linear(1 * 32 * 32, num_classes)  # Assuming input images are 64x64 pixels

    def forward(self, x):
        x = x.float()  # Convert input to float
        x = F.relu(self.conv1(x))  # Apply convolution, then ReLU activation
        x = x.view(-1, 1 * 32 * 32)  # Flatten the output for the dense layer
        x = self.fc1(x)  # Final dense layer to produce class scores
        return x

model = ExampleCNN(unzip_train_dataloader.dataset.num_classes)
model.to('cpu')
# Create the model object
modlee_model = ModleeImageClassifier(model=model)


modlee.mlflow.end_run()

trainer = trainer_for_modality_task(
    modality=exercise_modality,
    task=exercise_task,
    )

trainer.dataloader = unzip_train_dataloader
trainer.model = modlee_model

trainer.train(max_epochs=1, val_dataloaders=unzip_val_dataloader)

submit(api_key,exercise_id,trainer.model,example_batch_images,modlee)
print('As a reminder, your exercises model_size_restriction_MB is ',model_size_restriction_MB)
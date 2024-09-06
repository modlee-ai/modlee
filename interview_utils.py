import os
import zipfile
import shutil
import requests
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.jit as jit
import torch.nn.functional as F
import lightning.pytorch as pl
import pandas as pd
import numpy as np
import json
from torchvision import transforms
from PIL import Image

download_dir = './modlee_interview_data'
root_url = 'https://evalserver.modlee.ai:6060'

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.targets = pd.read_csv(os.path.join(data_dir, 'targets.csv')).values.squeeze()
        self.image_files = [os.path.join(data_dir, f'image_{i}.jpeg') for i in range(len(self.targets))]
        self.num_classes = len(np.unique(self.targets))
        self.classes = np.unique(self.targets)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx])
        if self.transform:
            img = self.transform(img)
        target = self.targets[idx]
        return img, target

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def setup(api_key, exercise_id):
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    download_zip_file = os.path.join(download_dir, 'modlee_interview_data.zip')

    response = requests.get(f"{root_url}/get-exercise", params={'api_key': api_key, 'exercise_id': exercise_id})

    if response.status_code == 200:
        with open(download_zip_file, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully: {download_zip_file}")
    else:
        print(f"Failed to download file: {response.status_code} {response.reason}")

    with zipfile.ZipFile(download_zip_file, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    interview_folder = [f'{download_dir}/{f}' for f in os.listdir(download_dir) if not f.endswith('.zip')][0]

    train_folder = f'{interview_folder}/train'
    val_folder = f'{interview_folder}/val'

    if exercise_id.startswith('TS-'):  # Time series task
        with open(os.path.join(interview_folder, 'shapes.json'), 'r') as f:
            shapes = json.load(f)

        X_train = pd.read_csv(f'{train_folder}/X_train.csv').values
        y_train = pd.read_csv(f'{train_folder}/y_train.csv').values
        X_val = pd.read_csv(f'{val_folder}/X_val.csv').values
        y_val = pd.read_csv(f'{val_folder}/y_val.csv').values

        X_train = X_train.reshape(shapes['X_train'])
        y_train = y_train.reshape(shapes['y_train'])
        X_val = X_val.reshape(shapes['X_val'])
        y_val = y_val.reshape(shapes['y_val'])

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        return train_dataloader, val_dataloader, X_train.shape, y_train.shape
    
    else:  # Image classification task
        train_dataset = CustomDataset(data_dir=train_folder, transform=train_transform)
        val_dataset = CustomDataset(data_dir=val_folder, transform=val_transform)
        
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        for _, data in enumerate(train_dataloader, 0):
            example_batch, _ = data
            break
        
        return train_dataloader, val_dataloader, example_batch


def submit(api_key, exercise_id, model, example_batch, modlee):
    logs = {}

    if exercise_id.startswith('TS-'):  # Time series task
        traced_model = jit.script(model)
    else:  # Image classification task
        last_run_path = modlee.last_run_path()
        print(f"Modlee last run path: {last_run_path}")

        artifacts_path = os.path.join(last_run_path, 'artifacts')
        artifacts_submit = ['model_graph.py', 'model_graph.txt', 'stats_rep']

        for artifact in artifacts_submit:
            with open(os.path.join(artifacts_path, artifact), 'r') as file:
                logs[artifact] = file.read()

        metrics_path = os.path.join(last_run_path, 'metrics')
        metrics_submit = ['val_loss']

        for metric in metrics_submit:
            with open(os.path.join(metrics_path, metric), 'r') as file:
                logs[metric] = file.read()
        traced_model = jit.trace(model, example_batch)
    

    saved_model_path = f'{download_dir}/traced_model.pt'
    traced_model.save(saved_model_path)

    url = f"{root_url}/evaluate-model"
    files = {'file': (saved_model_path, open(saved_model_path, 'rb'), 'application/octet-stream')}
    data = {'api_key': api_key, 'exercise_id': exercise_id}
    data.update(logs)

    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        print("Request was successful.")
        print("Response:", response.json())
    else:
        print("Request failed with status code:", response.status_code)
        print("Response:", response.text)

import modlee

class ModleeImageClassifier(modlee.model.ModleeModel):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.loss_fn = F.cross_entropy

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
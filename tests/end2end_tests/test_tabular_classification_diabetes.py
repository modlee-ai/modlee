import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
import pytest
from utils import check_artifacts
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

device = torch.device('cpu')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert features to tensors
        self.y = torch.tensor(y, dtype=torch.long) # Convert labels to long integers for classification

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_diabetes_dataloaders(batch_size=32, val_split=0.2, shuffle=True):
    # Load the diabetes dataset from sklearn
    diabetes = load_diabetes()
    X = diabetes.data  # Features
    y = diabetes.target  # Labels (for regression)

    # Determine the number of features and classes
    num_features = X.shape[1]  # Number of features (columns in X)
    
    # Since this dataset is for regression, the number of classes could be considered 1 (continuous target).
    # If you intend to convert it into a classification problem, you'd have to binarize or categorize y.
    num_classes = len(np.unique(y))  # Unique values in y (though it's for regression, this will count distinct targets)

    # Initialize the scaler for feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scale the features

    # Create a TabularDataset instance
    dataset = TabularDataset(X_scaled, y)

    # Split the dataset into training and validation sets
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader instances for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader, num_features, num_classes


class TabularClassifier(modlee.model.TabularClassificationModleeModel):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def test_tabular_classifier():    

    train_dataloader, val_dataloader, num_features, num_classes = get_diabetes_dataloaders()

    modlee_model = TabularClassifier(input_dim=num_features, num_classes=num_classes).to(device)

    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":

    test_tabular_classifier()
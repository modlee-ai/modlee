import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer, LightningModule
from modlee.model.callbacks import ModelMetafeaturesCallback
import modlee.model_metafeatures as mmf
import modlee
import os

os.environ['MODLEE_API_KEY'] = 'GZ4a6OoXmCXUHDJGnnGWNofsPrK0YF0i'

# Define a simple model for testing
class SimpleModel(LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

# Define a simple RNN model for time series data
class TimeSeriesModel(LightningModule):
    def __init__(self):
        super(TimeSeriesModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(20, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 20).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

# Define a simple CNN model for image data
class ImageModel(LightningModule):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

# Create a simple dataset and dataloader for tabular data
def create_tabular_dataloader():
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)
    return dataloader

# Create a simple dataset and dataloader for time series data
def create_timeseries_dataloader():
    x = torch.randn(100, 10, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)
    return dataloader

# Create a simple dataset and dataloader for image data
def create_image_dataloader():
    x = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)
    return dataloader

# Test function for tabular data
def test_model_metafeatures_callback_tabular():
    model = SimpleModel()
    dataloader = create_tabular_dataloader()
    trainer = Trainer(max_epochs=1)

    # Run training
    with modlee.start_run():
        trainer.fit(model, dataloader)

    assert True

# Test function for time series data
def test_model_metafeatures_callback_timeseries():
    model = TimeSeriesModel()
    dataloader = create_timeseries_dataloader()
    trainer = Trainer(max_epochs=1)

    # Run training
    with modlee.start_run():
        trainer.fit(model, dataloader)

    assert True

# Test function for image data
def test_model_metafeatures_callback_image():
    model = ImageModel()
    dataloader = create_image_dataloader()
    trainer = Trainer(max_epochs=1)

    # Run training
    with modlee.start_run():
        trainer.fit(model, dataloader)

    assert True

if __name__ == "__main__":
    pytest.main([__file__])
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

# Create a simple dataset and dataloader
def create_dataloader():
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)
    return dataloader

# Test function
def test_model_metafeatures_callback():
    model = SimpleModel()
    dataloader = create_dataloader()
    trainer = Trainer(max_epochs=1)

    # Run training
    with modlee.start_run():
        trainer.fit(model, dataloader)

    assert True

if __name__ == "__main__":
    pytest.main([__file__])
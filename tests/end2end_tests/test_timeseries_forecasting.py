import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytest
from utils import check_artifacts, get_device

device = get_device()
modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

def generate_dummy_time_series_data(num_samples=1000, seq_length=20, num_features=10, output_features=5):
    X = torch.randn(num_samples, seq_length, num_features)
    y = torch.randn(num_samples, seq_length, output_features)
    return X, y

class MultivariateTimeSeriesForecaster(modlee.model.TimeseriesForecastingModleeModel):
    def __init__(self, input_dim, seq_length, output_dim, hidden_dim=64):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        # Neural network layers
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

        # Loss function (MSE for regression)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size, seq_length, input_dim = x.shape

        # Process each time step independently through the feedforward network
        x = x.view(-1, input_dim)  # Flatten time dimension with batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        # Reshape back to (batch_size, seq_length, output_dim)
        x = x.view(batch_size, seq_length, -1)
        return x

    def training_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Modify the parameter combinations to include output features
parameter_combinations = [
    (5, 3, 2),  # num_features, seq_length, output_features
    (10, 6, 5),
    (10, 30, 10),
    (100, 20, 50)
]

@pytest.mark.parametrize("num_features, seq_length, output_features", parameter_combinations)
def test_multivariate_time_series_forecaster(num_features, seq_length, output_features):
    X, y = generate_dummy_time_series_data(num_samples=1000, seq_length=seq_length, num_features=num_features, output_features=output_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Debugging: Print the first batch of the train dataloader
    for batch in train_dataloader:
        print(f"Train batch X shape: {batch[0].shape}, y shape: {batch[1].shape}")
        break

    # Debugging: Print the first batch of the test dataloader
    for batch in test_dataloader:
        print(f"Test batch X shape: {batch[0].shape}, y shape: {batch[1].shape}")
        break

    lightning_model = MultivariateTimeSeriesForecaster(input_dim=num_features, seq_length=seq_length, output_dim=output_features, hidden_dim=64).to(device)

    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(
            model=lightning_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_multivariate_time_series_forecaster(5, 10, 3)

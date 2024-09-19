import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytest

device = torch.device('cpu')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

def generate_dummy_time_series_data(num_samples=1000, seq_length=20, num_features=10):
    X = torch.randn(num_samples, seq_length, num_features)
    y = torch.randn(num_samples, seq_length)
    return X, y

class TimeSeriesForecaster(modlee.model.TimeseriesClassificationModleeModel):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :]) 
        return predictions

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

parameter_combinations = [
    (5, 10),
    (5, 20),
    (5, 30),
    (10, 10),
    (10, 20),
    (10, 30),
    (20, 10),
    (20, 20),
    (20, 30)
]

@pytest.mark.parametrize("num_features, seq_length", parameter_combinations)
def test_time_series_forecaster(num_features, seq_length):
    X, y = generate_dummy_time_series_data(num_samples=1000, seq_length=seq_length, num_features=num_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for batch in train_dataloader:
        print(f"Train batch X shape: {batch[0].shape}, y shape: {batch[1].shape}")
        break

    # Debugging: Print the first batch of the test dataloader
    for batch in test_dataloader:
        print(f"Test batch X shape: {batch[0].shape}, y shape: {batch[1].shape}")
        break

    lightning_model = TimeSeriesForecaster(input_dim=num_features, hidden_dim=64, output_dim=1).to(device)

    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(
            model=lightning_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    artifacts = os.listdir(artifacts_path)

    assert artifacts, f"No artifacts found in the path: {artifacts_path}"

    print(f"Run path: {last_run_path}")
    print(f"Saved artifacts: {artifacts}")

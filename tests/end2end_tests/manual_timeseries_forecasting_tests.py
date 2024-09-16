import torch
import os
import modlee
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

os.environ['MODLEE_API_KEY'] = "OktSzjtS27JkuFiqpuzzyZCORw88Cz0P"
modlee.init(api_key=os.environ.get('MODLEE_API_KEY'))

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class LightningTimeSeriesForecaster(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.modlee_model = TimeSeriesForecaster(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        return self.modlee_model(x)

    def training_step(self, batch, batch_idx):
        return self.modlee_model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.modlee_model.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.modlee_model.configure_optimizers()

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

def run_manual_tests():
    for num_features, seq_length in parameter_combinations:
        print(f"Testing with num_features={num_features} and seq_length={seq_length}")

        X, y = generate_dummy_time_series_data(num_samples=1000, seq_length=seq_length, num_features=num_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        lightning_model = LightningTimeSeriesForecaster(input_dim=num_features, hidden_dim=64, output_dim=1)

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
        assert artifacts, "No artifacts found in the artifacts path."

        print(f"Run path: {last_run_path}")
        print(f"Saved artifacts: {artifacts}")

if __name__ == "__main__":
    run_manual_tests()

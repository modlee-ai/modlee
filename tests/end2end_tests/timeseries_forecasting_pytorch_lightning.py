#pytorch_lightning version 
import torch
import os, sys
import modlee
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

os.environ['MODLEE_API_KEY'] = "OktSzjtS27JkuFiqpuzzyZCORw88Cz0P"
modlee.init(api_key=os.environ.get('MODLEE_API_KEY'))

def generate_dummy_time_series_data(num_samples=1000, seq_length=20, num_features=10):
    X = torch.randn(num_samples, seq_length, num_features) 
    y = torch.randn(num_samples, seq_length) 
    return X, y

X, y = generate_dummy_time_series_data(num_samples=1000, seq_length=20, num_features=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class TimeSeriesForecaster(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(TimeSeriesForecaster, self).__init__()
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

input_dim = X_train.shape[2]
time_series_model = TimeSeriesForecaster(input_dim=input_dim, hidden_dim=64, output_dim=1)

with modlee.start_run() as run:
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(
        model=time_series_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader
    )

last_run_path = modlee.last_run_path()
print(f"Run path: {last_run_path}")

artifacts_path = os.path.join(last_run_path, 'artifacts')
artifacts = os.listdir(artifacts_path)
print(f"Saved artifacts: {artifacts}")

os.environ['ARTIFACTS_PATH'] = artifacts_path
sys.path.insert(0, artifacts_path)

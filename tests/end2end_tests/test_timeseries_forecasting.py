
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
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape
        x = x.view(-1, input_dim)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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

class TransformerTimeSeriesForecaster(modlee.model.TimeseriesForecastingModleeModel):
    def __init__(self, input_dim, seq_length, output_dim, nhead=2, num_layers=2):
        super().__init__()
        self.seq_length = seq_length
        self.transformer = torch.nn.Transformer(input_dim, nhead=nhead, num_encoder_layers=num_layers)
        self.fc_out = torch.nn.Linear(input_dim, output_dim)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = x.permute(1, 0, 2)  
        x = self.transformer(x, x)
        x = self.fc_out(x.permute(1, 0, 2)) 
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

parameter_combinations = [
    (5, 3, 2),
    (10, 6, 5),
    (10, 30, 10),
    (100, 20, 50)
]

model_classes = ['multivariate', 'transformer']

@pytest.mark.parametrize("model_class", model_classes)
@pytest.mark.parametrize("num_features, seq_length, output_features", parameter_combinations)
def test_multivariate_time_series_forecaster(model_class, num_features, seq_length, output_features):
    X, y = generate_dummy_time_series_data(num_samples=1000, seq_length=seq_length, num_features=num_features, output_features=output_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if model_class == 'multivariate':
        model = TransformerTimeSeriesForecaster(input_dim=num_features, seq_length=seq_length, output_dim=output_features, nhead=1).to(device)
    else:
        model = MultivariateTimeSeriesForecaster(input_dim=num_features, seq_length=seq_length, output_dim=output_features).to(device)

    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_multivariate_time_series_forecaster(MultivariateTimeSeriesForecaster, 5, 10, 3)
    test_multivariate_time_series_forecaster(TransformerTimeSeriesForecaster, 5, 10, 3)

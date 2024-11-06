
import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
import pytest
import pandas as pd
from utils import check_artifacts
from sklearn.model_selection import train_test_split

device = torch.device('cpu')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

def load_shampoo_data(file_path, seq_length):
    data = pd.read_csv(file_path)
    data['Month'] = pd.to_datetime(data['Month'], format='%d-%b')  
    data.set_index('Month', inplace=True)

    y = data['Sales of shampoo over a three year period'].values  
    y = torch.tensor(y, dtype=torch.float32)

    num_samples = len(y) - seq_length + 1
    y_seq = torch.stack([y[i:i + seq_length] for i in range(num_samples)])

    X_seq = torch.zeros(num_samples, seq_length, 1) 

    return X_seq, y_seq

def load_stock_data(file_path, seq_length):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    X = data[['Open', 'High', 'Low', 'Volume']].values  
    y = data['Close'].values  
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    num_samples = X.shape[0] - seq_length + 1
    X_seq = torch.stack([X[i:i + seq_length] for i in range(num_samples)])
    y_seq = y[seq_length - 1:]  

    return X_seq, y_seq

def load_power_consumption_data(file_path, seq_length):
    data = pd.read_csv(file_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)
    
    X = data[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']].values
    y = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']].mean(axis=1).values  
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    num_samples = X.shape[0] - seq_length + 1
    X_seq = torch.stack([X[i:i + seq_length] for i in range(num_samples)])
    y_seq = y[seq_length - 1:]  

    return X_seq, y_seq

class MultivariateTimeSeriesRegressor(modlee.model.TimeseriesRegressionModleeModel):
    def __init__(self, input_dim, seq_length, hidden_dim=64):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        self.fc1 = torch.nn.Linear(input_dim * seq_length, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape
        x = x.view(batch_size, -1)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x 

    def training_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = torch.nn.functional.mse_loss(preds, y) 
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self.forward(x)
        loss = torch.nn.functional.mse_loss(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

recommended_model_list = [False]
modlee_trainer_list = [True, False]

@pytest.mark.parametrize("input_dim, seq_length, dataset_type", [
    (4, 3, 'stock'),
    (4, 6, 'stock'),
    (4, 30, 'stock'),
    (4, 20, 'stock'),
    (5, 3, 'power_consumption'),
    (5, 6, 'power_consumption'),
    (5, 30, 'power_consumption'),
    (5, 20, 'power_consumption'),
    (1, 3, 'shampoo'),  
    (1, 6, 'shampoo'),
    (1, 30, 'shampoo'),
    (1, 20, 'shampoo')
])
@pytest.mark.parametrize("recommended_model", recommended_model_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
def test_time_series_regression(input_dim, seq_length, dataset_type, recommended_model, modlee_trainer):
    if dataset_type == 'stock':
        file_path = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data'), 'A.csv')
        print(file_path)
        X, y = load_stock_data(file_path, seq_length)
    elif dataset_type == 'power_consumption':
        file_path = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data'), 'powerconsumption.csv')
        X, y = load_power_consumption_data(file_path, seq_length)
    else:  
        file_path = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data'), 'sales-of-shampoo-over-a-three-ye.csv')
        X, y = load_shampoo_data(file_path, seq_length)

    dataset = TensorDataset(X, y)
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X[train_indices], y[train_indices])
    val_dataset = TensorDataset(X[val_indices], y[val_indices])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    if recommended_model:
        recommender = modlee.recommender.MultivariateTimeSeriesRegressor()  # Placeholder for actual recommender
        recommender.fit(train_dataloader)
        modlee_model = recommender.model
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        model = MultivariateTimeSeriesRegressor(input_dim=input_dim, seq_length=seq_length).to(device)
    
    if modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=1)
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader 
            )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

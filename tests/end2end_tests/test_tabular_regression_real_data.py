import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
from sklearn.model_selection import train_test_split
import pytest
from utils import check_artifacts

device = torch.device('mps')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

def load_california_housing_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

def load_diabetes_data():
    data = load_diabetes()
    X, y = data.data, data.target
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

def load_wine_quality_data():
    data = fetch_openml(name="wine-quality-red", version=1)
    X = data.data.to_numpy()  
    y = data.target.astype(float).to_numpy()  
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

class TabularRegression(modlee.model.TabularRegressionModleeModel):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        preds = self.forward(x).squeeze() 
        loss = self.loss_fn(preds, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        preds = self.forward(x).squeeze()
        loss = self.loss_fn(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

@pytest.mark.parametrize("load_data_func", [
    load_california_housing_data,  
    load_diabetes_data,           
    load_wine_quality_data       
])

def test_tabular_regressor(load_data_func):
    X, y = load_data_func()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    modlee_model = TabularRegression(input_dim=X_train.shape[1]).to(device)

    trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
    trainer.fit(
        model=modlee_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader
    )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":
    test_tabular_regressor(load_california_housing_data)

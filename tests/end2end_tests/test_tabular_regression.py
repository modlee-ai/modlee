import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytest
from utils import check_artifacts

device = torch.device('mps')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

def generate_dummy_tabular_data_regression(num_samples=100, num_features=10):
    X = torch.randn(num_samples, num_features)
    y = torch.randn(num_samples) 
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

num_samples_list = [100, 200, 300]
num_features_list = [10, 20, 50, 100]

@pytest.mark.parametrize("num_samples", num_samples_list)
@pytest.mark.parametrize("num_features", num_features_list)
def test_tabular_regressor(num_samples, num_features):    
    X, y = generate_dummy_tabular_data_regression(num_samples=num_samples, num_features=num_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    modlee_model = TabularRegression(input_dim=num_features).to(device)

    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

if __name__ == "__main__":

    test_tabular_regressor(100, 10)

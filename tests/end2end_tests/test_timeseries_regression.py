import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytest
from utils import check_artifacts

device = torch.device('cpu')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

def generate_dummy_time_series_data(num_samples=1000, seq_length=20, num_features=10):
    X = torch.randn(num_samples, seq_length, num_features)
    y = torch.randn(num_samples, 1)  
    return X, y

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

parameter_combinations = [
    (5, 3), 
    (10, 6),
    (10, 30),
    (100, 20)
]
recommended_model_list = [False]
modlee_trainer_list = [True, False]

@pytest.mark.parametrize("num_features, seq_length", parameter_combinations)
@pytest.mark.parametrize("recommended_model", recommended_model_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
def test_multivariate_time_series_regressor(num_features, seq_length, recommended_model, modlee_trainer):
    X, y = generate_dummy_time_series_data(num_samples=1000, seq_length=seq_length, num_features=num_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    if recommended_model:
        recommender = modlee.recommender.TimeSeriesRegressionRecommender()  # Placeholder for actual recommender
        recommender.fit(train_dataloader)
        modlee_model = recommender.model
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        modlee_model = MultivariateTimeSeriesRegressor(input_dim=num_features, seq_length=seq_length).to(device)

    if modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )
    else:
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
    test_multivariate_time_series_regressor(5, 20, False, True)

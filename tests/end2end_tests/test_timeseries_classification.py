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

def generate_dummy_time_series_data(num_samples=1000, seq_length=20, num_features=10, num_classes=5):
    X = torch.randn(num_samples, seq_length, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

class MultivariateTimeSeriesClassifier(modlee.model.TimeseriesClassificationModleeModel):
    def __init__(self, input_dim, seq_length, num_classes, hidden_dim=64):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        self.fc1 = torch.nn.Linear(input_dim * seq_length, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape
        x = x.view(batch_size, -1) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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

class SimpleTransformerModel(modlee.model.TimeseriesClassificationModleeModel):
    def __init__(self, input_dim, seq_length, num_classes, nhead=2, d_model=32):
        super().__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.nhead = nhead

        if input_dim != d_model:
            self.input_proj = torch.nn.Linear(input_dim, d_model)
        else:
            self.input_proj = None
        self.d_model = d_model

        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=2
        )
        self.fc = torch.nn.Linear(seq_length * d_model, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        if self.input_proj:
            x = self.input_proj(x)
        x = self.transformer_encoder(x.permute(1, 0, 2))  
        x = x.permute(1, 0, 2).contiguous()  
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
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
    (5, 3, 3), 
    (10, 6, 5),
    (10, 30, 10),
    (100, 20, 50)
]
recommended_model_list = [False]
modlee_trainer_list = [True, False]
model_types = ['multivariate', 'transformer']  

@pytest.mark.parametrize("num_features, seq_length, num_classes", parameter_combinations)
@pytest.mark.parametrize("recommended_model", recommended_model_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
@pytest.mark.parametrize("model_type", model_types)  
def test_multivariate_time_series_classifier(num_features, seq_length, num_classes, recommended_model, modlee_trainer, model_type):
    X, y = generate_dummy_time_series_data(num_samples=1000, seq_length=seq_length, num_features=num_features, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    if recommended_model:
        recommender = modlee.recommender.TimeSeriesClassificationRecommender()  # Placeholder for actual recommender
        recommender.fit(train_dataloader)
        modlee_model = recommender.model
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        if model_type == 'multivariate':
            modlee_model = MultivariateTimeSeriesClassifier(input_dim=num_features, seq_length=seq_length, num_classes=num_classes).to(device)
        else:
            modlee_model = SimpleTransformerModel(input_dim=num_features, seq_length=seq_length, num_classes=num_classes).to(device)

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
    test_multivariate_time_series_classifier(5, 20, 3, False, True)


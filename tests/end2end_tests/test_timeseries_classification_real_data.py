import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
import pytest
from utils import check_artifacts
import pandas as pd

device = torch.device('cpu')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

def load_ecg200_from_txt(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    y = data.iloc[:, 0].values  
    X = data.iloc[:, 1:].values  
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) 
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def load_beef_from_txt(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    y = data.iloc[:, 0].values 
    X = data.iloc[:, 1:].values 
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def load_car_from_txt(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    y = data.iloc[:, 0].values  
    X = data.iloc[:, 1:].values  
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  
    y = torch.tensor(y, dtype=torch.long)
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

class TransformerTimeSeriesClassifier(modlee.model.TimeseriesClassificationModleeModel):
    def __init__(self, input_dim, seq_length, num_classes, num_heads=1, hidden_dim=64):
        super().__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = torch.nn.Linear(input_dim * seq_length, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.transformer_encoder(x)
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

recommended_model_list = [False]
modlee_trainer_list = [True, False]

@pytest.mark.parametrize("num_features, seq_length, num_classes, dataset_type", [
    (1,96,2,'ecg'),
    (1, 96, 6, 'ecg'),      
    (1, 470, 5,'beef'),
    (1, 470, 10, 'beef'),      
    (1, 577, 4, 'car'),
    (1, 577, 8, 'car'),
])
@pytest.mark.parametrize("recommended_model", recommended_model_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
@pytest.mark.parametrize("model_type", ['transformer', 'multivariate'])
def test_classifier(num_features, seq_length, num_classes, dataset_type, recommended_model, modlee_trainer, model_type):
    
    if dataset_type == 'ecg':
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'ECG200_TRAIN.txt')
        X_train, y_train = load_ecg200_from_txt(file_path)
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'ECG200_TEST.txt')
        X_test, y_test = load_ecg200_from_txt(file_path)

    elif dataset_type == 'beef':
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'Beef_TRAIN.txt')
        X_train, y_train = load_beef_from_txt(file_path)
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'Beef_TEST.txt')
        X_test, y_test = load_beef_from_txt(file_path)

    else:
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'Car_TRAIN.txt')
        X_train, y_train = load_car_from_txt(file_path)
        file_path = os.path.join(os.getcwd(), 'tests/end2end_tests/time_series_data', 'Car_TEST.txt')
        X_test, y_test = load_car_from_txt(file_path)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    if recommended_model:
        recommender = modlee.recommender.MultivariateTimeSeriesClassifier()  
        recommender.fit(train_dataloader)
        modlee_model = recommender.model
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        if model_type == 'multivariate':
            modlee_model = MultivariateTimeSeriesClassifier(input_dim=num_features, seq_length=seq_length, num_classes=num_classes).to(device)
        elif model_type == 'transformer':
            modlee_model = TransformerTimeSeriesClassifier(input_dim=num_features, seq_length=seq_length, num_classes=num_classes).to(device)

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

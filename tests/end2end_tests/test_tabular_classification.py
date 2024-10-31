import torch
import os
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytest
from utils import check_artifacts

device = torch.device('cpu')
#modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')
modlee.init(api_key='kF4dN7mP9qW2sT8v', run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

def generate_dummy_tabular_data(num_samples=100, num_features=10, num_classes=2):
    """Generate dummy tabular data."""
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

class TabularClassifier(modlee.model.TabularClassificationModleeModel):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

num_samples_list = [100]
num_features_list = [20,100]
num_classes_list = [2,10]
recommended_model_list = [True,False]
modlee_trainer_list = [True,False]

@pytest.mark.parametrize("num_samples", num_samples_list)
@pytest.mark.parametrize("num_features", num_features_list)
@pytest.mark.parametrize("num_classes", num_classes_list)
@pytest.mark.parametrize("recommended_model", recommended_model_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)

def test_tabular_classifier(num_samples, num_features, num_classes, recommended_model, modlee_trainer):
    X, y = generate_dummy_tabular_data(num_samples=num_samples, num_features=num_features, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    #train_dataloader, val_dataloader, num_features, num_classes = get_diabetes_dataloaders()

    if recommended_model == True:
        recommender = modlee.recommender.TabularClassificationRecommender(num_classes=num_classes)
        recommender.fit(train_dataloader)
        modlee_model = recommender.model.to(device)        
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        modlee_model = TabularClassifier(input_dim=num_features, num_classes=num_classes).to(device)    
    # modlee_trainer = True
    if modlee_trainer == True:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=10)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=10)
            trainer.fit(
                model=modlee_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader
            )


    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)


if __name__ == "__main__":

    test_tabular_classifier(100, 10, 2, True, True)
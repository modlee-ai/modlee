import torch
import os
import modlee
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

os.environ['MODLEE_API_KEY'] = "OktSzjtS27JkuFiqpuzzyZCORw88Cz0P"
modlee.init(api_key=os.environ.get('MODLEE_API_KEY'))

def generate_dummy_tabular_data(num_samples=100, num_features=10, num_classes=2):
    """Generate dummy tabular data."""
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

class TabularClassifier(modlee.model.TabularClassificationModleeModel):
    def __init__(self, input_dim, num_classes=2):
        super().__init__(input_dim=input_dim)
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

class LightningTabularClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.modlee_model = TabularClassifier(input_dim=input_dim, num_classes=num_classes)

    def forward(self, x):
        return self.modlee_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.modlee_model(x)
        loss = self.modlee_model.loss_fn(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.modlee_model(x)
        loss = self.modlee_model.loss_fn(logits, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.modlee_model.parameters(), lr=1e-3)

def manual_test():
    """Manually test the model training with all parameter combinations."""
    num_samples_list = [100, 500, 1000, 2000]
    num_features_list = [10, 20, 30, 40]
    num_classes_list = [2, 5, 10]

    for num_samples in num_samples_list:
        for num_features in num_features_list:
            for num_classes in num_classes_list:
                print(f"Testing with num_samples={num_samples}, num_features={num_features}, num_classes={num_classes}")
                
                X, y = generate_dummy_tabular_data(num_samples=num_samples, num_features=num_features, num_classes=num_classes)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                train_dataset = TensorDataset(X_train, y_train)
                test_dataset = TensorDataset(X_test, y_test)

                train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                lightning_model = LightningTabularClassifier(input_dim=num_features, num_classes=num_classes)

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
                print(f"Run path: {last_run_path}")
                print(f"Saved artifacts: {artifacts}")

if __name__ == "__main__":
    manual_test()

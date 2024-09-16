import torch
import os
import modlee
import pytest
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

os.environ['MODLEE_API_KEY'] = "OktSzjtS27JkuFiqpuzzyZCORw88Cz0P"
modlee.init(api_key=os.environ.get('MODLEE_API_KEY'))

def generate_dummy_data(num_samples=100, num_classes=2, img_size=(3, 32, 32)):
    X = torch.randn(num_samples, *img_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


class ModleeImageClassification(modlee.model.ImageClassificationModleeModel):
    def __init__(self, num_classes=2, img_size=(3, 32, 32)):
        super().__init__(num_classes=num_classes)
        input_dim = img_size[0] * img_size[1] * img_size[2]  
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, 128),  
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

class LightningImageClassification(pl.LightningModule):
    def __init__(self, num_classes=2, img_size=(3, 32, 32)):
        super().__init__()
        self.modlee_model = ModleeImageClassification(num_classes=num_classes, img_size=img_size)

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

@pytest.mark.parametrize("img_size", [(3, 32, 32), (3, 64, 64), (3, 128, 128), (1, 32, 32), 
                                      (4, 32, 32), (4, 64, 64), (5, 32, 32), (5, 128, 128), 
                                      (6, 128, 128), (6, 256, 256)])
def test_model_training(img_size):
    X_train, y_train = generate_dummy_data(num_samples=100, img_size=img_size)
    X_test, y_test = generate_dummy_data(num_samples=20, img_size=img_size)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    lightning_model = LightningImageClassification(num_classes=2, img_size=img_size)

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
    assert artifacts, "No artifacts found in the artifacts path."

    print(f"Run path: {last_run_path}")
    print(f"Saved artifacts: {artifacts}")


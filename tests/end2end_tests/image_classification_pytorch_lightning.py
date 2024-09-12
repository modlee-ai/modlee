#pytorch_lightning version 
import torch
import os, sys
import modlee
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger

os.environ['MODLEE_API_KEY'] = "OktSzjtS27JkuFiqpuzzyZCORw88Cz0P"
modlee.init(api_key=os.environ.get('MODLEE_API_KEY'))

def generate_dummy_data(num_samples=100, num_classes=2, img_size=(3, 32, 32)):
    X = torch.randn(num_samples, *img_size)  
    y = torch.randint(0, num_classes, (num_samples,))  
    return X, y

X_train, y_train = generate_dummy_data(num_samples=100)
X_test, y_test = generate_dummy_data(num_samples=20)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ImageClassification(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 32 * 32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

lightning_model = ImageClassification(num_classes=2)

with modlee.start_run() as run:
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader
    )

last_run_path = modlee.last_run_path()
print(f"Run path: {last_run_path}")

artifacts_path = os.path.join(last_run_path, 'artifacts')
artifacts = os.listdir(artifacts_path)
print(f"Saved artifacts: {artifacts}")

os.environ['ARTIFACTS_PATH'] = artifacts_path
sys.path.insert(0, artifacts_path)

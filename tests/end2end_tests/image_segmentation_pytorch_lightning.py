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

def generate_dummy_segmentation_data(num_samples=100, img_size=(3, 32, 32), mask_size=(32, 32)):
    X = torch.randn(num_samples, *img_size)
    y = torch.randint(0, 2, (num_samples, 1, *mask_size)) 
    return X, y

X_train, y_train = generate_dummy_segmentation_data(num_samples=100)
X_test, y_test = generate_dummy_segmentation_data(num_samples=20)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ImageSegmentation(pl.LightningModule):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, num_classes, kernel_size=1),  
            torch.nn.Upsample(size=(32, 32))  
        )
        
        self.loss_fn = torch.nn.BCEWithLogitsLoss()  

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float()) 
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

lightning_model = ImageSegmentation(num_classes=1)  

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

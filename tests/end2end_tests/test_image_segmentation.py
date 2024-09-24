import torch
import os, sys
import modlee
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger
import pytest
from utils import check_artifacts

device = torch.device('cpu')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

def generate_dummy_segmentation_data(num_samples=100, img_size=(3, 32, 32), mask_size=(32, 32)):
    X = torch.randn(num_samples, *img_size)
    y = torch.randint(0, 2, (num_samples, 1, *mask_size))
    return X, y

class ImageSegmentation(modlee.model.ImageSegmentationModleeModel):
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=1),
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()  

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output_size = x.shape[2:]  
        decoded = torch.nn.functional.interpolate(decoded, size=output_size, mode='bilinear', align_corners=False)
        return decoded

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.float())
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
@pytest.mark.parametrize("img_size, mask_size", [
    ((1, 32, 32), (32, 32)),
    ((3, 32, 32), (32, 32)),
    ((4, 16, 16), (16, 16)),
    ((5, 128, 128), (128, 128)),
])
def test_segmentation_model_training(img_size, mask_size):
    X_train, y_train = generate_dummy_segmentation_data(num_samples=100, img_size=img_size, mask_size=mask_size)
    X_test, y_test = generate_dummy_segmentation_data(num_samples=20, img_size=img_size, mask_size=mask_size)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    in_channels = img_size[0] 
    lightning_model = ImageSegmentation(in_channels=in_channels).to(device)

    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(
            model=lightning_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)


if __name__ == "__main__":

    test_segmentation_model_training((3, 32, 32),(32, 32))
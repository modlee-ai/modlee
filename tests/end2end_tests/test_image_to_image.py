import torch
import os
import modlee
import pytest
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from utils import check_artifacts

device = torch.device('cpu')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

def generate_dummy_image_data(num_samples=100, img_size=(3, 32, 32)):
    X = torch.randn(num_samples, *img_size, device=device, dtype=torch.float32)
    y = torch.randn(num_samples, *img_size, device=device, dtype=torch.float32)
    return X, y

class ModleeImageToImageModel(modlee.model.ImageImageToImageModleeModel):
    def __init__(self, img_size=(3, 32, 32)):
        super().__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Conv2d(img_size[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, img_size[0], kernel_size=3, stride=1, padding=1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return {'loss': loss}
    
    def validation_step(self, val_batch):
        x, y_target = val_batch
        y_pred = self.forward(x)
        val_loss = self.loss_fn(y_pred, y_target)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

@pytest.mark.parametrize("num_samples", [100])
@pytest.mark.parametrize("img_size", [(3, 32, 32), (1, 28, 28), (3, 64, 64), (6, 128, 128)])
@pytest.mark.parametrize("recommended_model", [False])
@pytest.mark.parametrize("modlee_trainer", [True, False])
def test_image_to_image_model_training(num_samples, img_size, recommended_model, modlee_trainer):
    
    X_train, y_train = generate_dummy_image_data(num_samples=num_samples, img_size=img_size)
    X_test, y_test = generate_dummy_image_data(num_samples=num_samples, img_size=img_size)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    if recommended_model:
        recommender = modlee.recommender.ImageToImageRecommender()
        recommender.fit(train_dataloader)
        modlee_model = recommender.model
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        modlee_model = ModleeImageToImageModel(img_size=img_size).to(device)

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
    test_image_to_image_model_training(100, (3, 32, 32), False, True)

import torch
import os
import modlee
import pytest
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from utils import check_artifacts

device = torch.device('cpu')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

def add_noise(img, noise_level=0.1):
    noise = torch.randn_like(img) * noise_level
    return torch.clamp(img + noise, 0., 1.)

class NoisyImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_level=0.1, img_size=(1, 32, 32)):
        self.dataset = dataset
        self.noise_level = noise_level
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        
        if img.size(0) != self.img_size[0]:
            if img.size(0) < self.img_size[0]:  
                img = img.repeat(self.img_size[0] // img.size(0), 1, 1) 
            else:  
                img = img[:self.img_size[0], :, :] 

        img = transforms.Resize((self.img_size[1], self.img_size[2]))(img)
        noisy_img = add_noise(img, self.noise_level)
        return noisy_img, img  

class ModleeDenoisingModel(modlee.model.ImageImageToImageModleeModel):
    def __init__(self, img_size=(1, 32, 32)):
        super().__init__()
        in_channels = img_size[0]  
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return {'loss': self.loss_fn(y_pred, y)}
    
    def validation_step(self, val_batch, batch_idx):
        x, y_target = val_batch
        y_pred = self.forward(x)
        return {'val_loss': self.loss_fn(y_pred, y_target)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

class AutoencoderDenoisingModel(modlee.model.ImageImageToImageModleeModel):
    def __init__(self, img_size=(1, 32, 32)):
        super().__init__()
        in_channels = img_size[0]
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, padding=1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return {'loss': self.loss_fn(y_pred, y)}
    
    def validation_step(self, val_batch, batch_idx):
        x, y_target = val_batch
        y_pred = self.forward(x)
        return {'val_loss': self.loss_fn(y_pred, y_target)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

class UNetDenoisingModel(modlee.model.ImageImageToImageModleeModel):
    def __init__(self, img_size=(1, 32, 32)):
        super().__init__()
        in_channels = img_size[0]
        self.model = nn.Sequential( 
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, in_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x) 
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return {'loss': self.loss_fn(y_pred, y)}
    
    def validation_step(self, val_batch, batch_idx):
        x, y_target = val_batch
        y_pred = self.forward(x)
        return {'val_loss': self.loss_fn(y_pred, y_target)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

class ResNetDenoisingModel(modlee.model.ImageImageToImageModleeModel):
    def __init__(self, img_size=(1, 32, 32)):
        super().__init__()
        in_channels = img_size[0]
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return {'loss': self.loss_fn(y_pred, y)}
    
    def validation_step(self, val_batch, batch_idx):
        x, y_target = val_batch
        y_pred = self.forward(x)
        return {'val_loss': self.loss_fn(y_pred, y_target)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

def load_dataset(dataset_name, img_size, noise_level):
    transform = transforms.Compose([
        transforms.Resize((img_size[1], img_size[2])),
        transforms.ToTensor()
    ])
    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    elif dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")

    train_noisy_dataset = NoisyImageDataset(train_dataset, noise_level=noise_level, img_size=img_size)
    test_noisy_dataset = NoisyImageDataset(test_dataset, noise_level=noise_level, img_size=img_size)

    return train_noisy_dataset, test_noisy_dataset

model_classes = [AutoencoderDenoisingModel, UNetDenoisingModel, ResNetDenoisingModel, ModleeDenoisingModel]

@pytest.mark.parametrize("noise_level", [0.1])
@pytest.mark.parametrize("img_size", [(1, 32, 32), (3, 28, 28), (6, 28, 28)])
@pytest.mark.parametrize("recommended_model", [False])
@pytest.mark.parametrize("modlee_trainer", [True, False])
@pytest.mark.parametrize("dataset_name", ["CIFAR10", "MNIST", "FashionMNIST"])
@pytest.mark.parametrize("model_class", model_classes)
def test_denoising_model_training(noise_level, img_size, recommended_model, modlee_trainer, dataset_name, model_class):
    train_noisy_dataset, test_noisy_dataset = load_dataset(dataset_name, img_size, noise_level)
    train_dataloader = DataLoader(train_noisy_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_noisy_dataset, batch_size=32, shuffle=False)

    if recommended_model:
        recommender = modlee.recommender.ModleeDenoisingModel()  
        recommender.fit(train_dataloader)
        modlee_model = recommender.model
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        denoising_model_instance = model_class(img_size=img_size).to(device)

    if modlee_trainer:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
        trainer.fit(
            model=denoising_model_instance,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=1)
            trainer.fit(
                model=denoising_model_instance,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader
            )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)


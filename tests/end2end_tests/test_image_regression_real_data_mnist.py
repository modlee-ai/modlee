import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader
from torch import nn
import modlee
import lightning.pytorch as pl
from utils import check_artifacts

modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
])

def load_dataset(dataset_name='mnist'):
    if dataset_name == 'mnist':
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'fashion_mnist':
        train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    return train_dataloader, test_dataloader

class ModleeImageRegression(modlee.model.ImageRegressionModleeModel):
    def __init__(self, img_size=(1, 32, 32)):
        super().__init__()
        self.img_size = img_size
        input_dim = img_size[0] * img_size[1] * img_size[2]

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  
        )

        self.loss_fn = nn.MSELoss()  

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.forward(x).squeeze(), y.float())
        return {'loss': loss}

    def validation_step(self, val_batch):
        x, y_target = val_batch
        y_pred = self.forward(x)
        val_loss = self.loss_fn(y_pred.squeeze(), y_target.float())
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

def train_model(dataset_name='mnist'):
    train_dataloader, test_dataloader = load_dataset(dataset_name)
    
    modlee_model = ModleeImageRegression(img_size=(1, 32, 32)).to(torch.device('cpu'))

    trainer = modlee.model.trainer.AutoTrainer(max_epochs=1)
    trainer.fit(
        model=modlee_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader
    )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)

    print(f"{dataset_name} Training complete")

train_model('mnist')
train_model('fashion_mnist')
import torch
import os
import modlee
import pytest
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms
from PIL import Image
from utils import check_artifacts
import pandas as pd

device = torch.device('cpu')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

AGE_GROUPS = {'YOUNG': 0, 'MIDDLE': 1, 'OLD': 2}

class AgeDataset(Dataset):
    def __init__(self, image_folder, csv_file, img_size, transform=None):
        self.image_folder = image_folder
        self.data = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size[1:]),  
            transforms.ToTensor(),
        ])
        if 'Class' in self.data.columns:
            self.data['Class'] = self.data['Class'].map(AGE_GROUPS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

class PolygonDataset(Dataset):
    def __init__(self, image_folder, csv_file, img_size, transform=None):
        self.image_folder = image_folder
        self.data = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size[1:]), 
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.data.iloc[idx]['filename'])
        
        label = torch.tensor([
            self.data.iloc[idx]['bound_circle_x'], 
            self.data.iloc[idx]['bound_circle_y'], 
            self.data.iloc[idx]['bound_circle_r']
        ], dtype=torch.float32)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

class ModleeImageRegression(modlee.model.ImageRegressionModleeModel):
    def __init__(self, img_size=(3, 32, 32)):
        super().__init__()
        self.img_size = img_size
        self.input_dim = img_size[0] * img_size[1] * img_size[2]
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return {'loss': loss}
    
    def validation_step(self, val_batch):
        x, y_target = val_batch
        y_pred = self.forward(x)
        val_loss = self.loss_fn(y_pred, y_target)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

class BeautyRatingDataset(Dataset):
    def __init__(self, image_folder, txt_file, img_size, transform=None):
        self.image_folder = image_folder
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size[1:]),  
            transforms.ToTensor(),
        ])
        
        self.data = self.load_data(txt_file)

    def load_data(self, txt_file):
        data = []
        with open(txt_file, 'r') as file:
            for line in file:
                filename, rating = line.strip().split()
                data.append((filename, float(rating)))  
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.data[idx][0])
        label = torch.tensor(self.data[idx][1], dtype=torch.float32)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

age_image_folder = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'age')
age_csv_file = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'age.csv')
polygon_image_folder = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'polygon')
polygon_csv_file = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'polygon.csv')
beauty_image_folder = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'facial_rating')
beauty_txt_file = os.path.join(os.path.join(os.getcwd(), 'tests/end2end_tests/image_regression_data'), 'facial_rating.txt')

@pytest.mark.parametrize("img_size", [(3, 32, 32), (3, 64, 64), (3, 128, 128)])
@pytest.mark.parametrize("recommended_model", [False])
@pytest.mark.parametrize("modlee_trainer", [True, False])
@pytest.mark.parametrize("dataset", ["age", "polygon", "beauty"])  
def test_model_training(img_size, recommended_model, modlee_trainer, dataset):

    if dataset == "age":
        train_dataset = AgeDataset(age_image_folder, age_csv_file, img_size=img_size)
    elif dataset == "polygon":
        train_dataset = PolygonDataset(polygon_image_folder, polygon_csv_file, img_size=img_size)
    elif dataset == "beauty":
        train_dataset = BeautyRatingDataset(beauty_image_folder, beauty_txt_file, img_size=img_size)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    if recommended_model:
        recommender = modlee.recommender.ImageRegressionRecommender() 
        recommender.fit(train_dataloader)
        modlee_model = recommender.model
    else:
        modlee_model = ModleeImageRegression(img_size=img_size).to(device)

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
    test_model_training(img_size=(3, 32, 32), recommended_model=False, modlee_trainer=True, dataset="age")


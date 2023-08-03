#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

import os 
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
mps = torch.device('mps')
# torch.set_default_device(mps)
#%%
import mlflow
mlflow.get_tracking_uri()
#%%
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LitClf(pl.LightningModule):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        y_out = self.classifier(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        # loss = torch.nn.CrossEntropyLoss(y_out,y)
        loss = F.cross_entropy(y_out,y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.001,
            momentum=0.9
        )
        return optimizer
    
model = GarmentClassifier()
lit_model = LitClf(model)
# model.to(mps)
   
#%% 
# input_size, hidden_size, output_size = 10,20,5
# model = SimpleModel(input_size, hidden_size, output_size)

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)
#%%
bs = 4
training_loader = DataLoader(
    training_data, batch_size=bs,shuffle=True,
)
test_loader = DataLoader(
    test_data, batch_size=bs,shuffle=True,
)

#%%
mlflow.set_tracking_uri(
    'http://127.0.0.1:5000'
)
mlflow.get_tracking_uri()

#%%
mlflow.pytorch.autolog()
trainer = pl.Trainer()
trainer.fit(model=lit_model,
    train_dataloaders=training_loader,
    val_dataloaders=test_loader)
# breakpoint()


# # %%
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(
#     lit_model.parameters(),
#     lr=0.001,
#     momentum=0.9
# )
# # %%
# n_epochs = 10
# for epoch in range(n_epochs):
#     for train_batch in training_loader:
#         x_train,y_train = train_batch
#         x_train.to(mps), y_train.to(mps)
        
#         optimizer.zero_grad()
        
#         y_out = model(x_train)
#         loss = loss_fn(y_out,y_train)
#         loss.backward()
        
#         optimizer.step()
        
        
# # %%

# %%

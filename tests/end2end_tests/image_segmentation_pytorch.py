#pytorch

import torch
import os
import sys
import modlee
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

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

class ImageSegmentation(nn.Module):
    def __init__(self, num_classes=1):
        super(ImageSegmentation, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),  
            nn.Upsample(size=(32, 32)) 
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = ImageSegmentation(num_classes=1)  
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()  

def train_step(model, dataloader, optimizer, loss_fn):
    model.train()
    inputs, targets = next(iter(dataloader))  
    inputs, targets = inputs.float(), targets.float()  
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")

def evaluate_model(model, dataloader, loss_fn):
    model.eval()
    inputs, targets = next(iter(dataloader)) 
    outputs = model(inputs)
    loss = loss_fn(outputs, targets.float())
    return loss.item()

train_step(model, train_dataloader, optimizer, loss_fn)

val_loss = evaluate_model(model, test_dataloader, loss_fn)
print(f"Validation Loss: {val_loss}")

last_run_path = modlee.last_run_path()
print(f"Run path: {last_run_path}")

artifacts_path = os.path.join(last_run_path, 'artifacts')
artifacts = os.listdir(artifacts_path)
print(f"Saved artifacts: {artifacts}")

os.environ['ARTIFACTS_PATH'] = artifacts_path
sys.path.insert(0, artifacts_path)


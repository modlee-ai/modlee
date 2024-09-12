#pytorch version
import torch
import os, sys
import modlee
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

os.environ['MODLEE_API_KEY'] = "OktSzjtS27JkuFiqpuzzyZCORw88Cz0P"
modlee.init(api_key=os.environ.get('MODLEE_API_KEY'))

def generate_dummy_tabular_data(num_samples=1000, num_features=10, num_classes=2):
    X = torch.randn(num_samples, num_features)  
    y = torch.randint(0, num_classes, (num_samples,))  
    return X, y

X, y = generate_dummy_tabular_data(num_samples=1000, num_features=10, num_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class TabularClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(TabularClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

input_dim = X_train.shape[1]
tabular_model = TabularClassifier(input_dim=input_dim, num_classes=2)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(tabular_model.parameters(), lr=1e-3)

def train_model_one_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")

def validate_model(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

with modlee.start_run() as run:
    train_model_one_epoch(tabular_model, train_dataloader, optimizer, loss_fn)
    validate_model(tabular_model, test_dataloader, loss_fn)

last_run_path = modlee.last_run_path()
print(f"Run path: {last_run_path}")

artifacts_path = os.path.join(last_run_path, 'artifacts')
artifacts = os.listdir(artifacts_path)
print(f"Saved artifacts: {artifacts}")

os.environ['ARTIFACTS_PATH'] = artifacts_path
sys.path.insert(0, artifacts_path)

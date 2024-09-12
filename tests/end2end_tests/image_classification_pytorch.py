#pytorch version 
import torch
import os, sys
import modlee
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

class ImageClassification(torch.nn.Module):
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

    def compute_loss(self, logits, y):
        return self.loss_fn(logits, y)

model = ImageClassification(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x, y = batch
        optimizer.zero_grad()
        logits = model(x)
        loss = model.compute_loss(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            logits = model(x)
            loss = model.compute_loss(logits, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

with modlee.start_run() as run:
    train_loss = train(model, train_dataloader, optimizer)
    val_loss = validate(model, test_dataloader)
    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

last_run_path = modlee.last_run_path()
print(f"Run path: {last_run_path}")

artifacts_path = os.path.join(last_run_path, 'artifacts')
artifacts = os.listdir(artifacts_path)
print(f"Saved artifacts: {artifacts}")

os.environ['ARTIFACTS_PATH'] = artifacts_path
sys.path.insert(0, artifacts_path)

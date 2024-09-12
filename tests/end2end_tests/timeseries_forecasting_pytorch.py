#pytorch version 
import torch
import os, sys
import modlee
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

os.environ['MODLEE_API_KEY'] = "OktSzjtS27JkuFiqpuzzyZCORw88Cz0P"
modlee.init(api_key=os.environ.get('MODLEE_API_KEY'))

def generate_dummy_time_series_data(num_samples=1000, seq_length=20, num_features=10):
    X = torch.randn(num_samples, seq_length, num_features)  
    y = torch.randn(num_samples, seq_length) 
    return X, y

X, y = generate_dummy_time_series_data(num_samples=1000, seq_length=20, num_features=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class TimeSeriesForecaster(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(TimeSeriesForecaster, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x) 
        predictions = self.fc(lstm_out[:, -1, :])  
        return predictions

input_dim = X_train.shape[2]
time_series_model = TimeSeriesForecaster(input_dim=input_dim, hidden_dim=64, output_dim=1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(time_series_model.parameters(), lr=1e-3)

def train_model(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x, y = batch
        optimizer.zero_grad() 
        preds = model(x) 
        loss = loss_fn(preds, y)  
        loss.backward()  
        optimizer.step()  
        total_loss += loss.item()
    print(f"Training Loss: {total_loss/len(train_loader)}")

def validate_model(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():  
        for batch in test_loader:
            x, y = batch
            preds = model(x) 
            loss = loss_fn(preds, y)  
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss/len(test_loader)}")

with modlee.start_run() as run:
    train_model(time_series_model, train_dataloader, optimizer, loss_fn)
    validate_model(time_series_model, test_dataloader, loss_fn)

last_run_path = modlee.last_run_path()
print(f"Run path: {last_run_path}")

artifacts_path = os.path.join(last_run_path, 'artifacts')
artifacts = os.listdir(artifacts_path)
print(f"Saved artifacts: {artifacts}")

os.environ['ARTIFACTS_PATH'] = artifacts_path
sys.path.insert(0, artifacts_path)

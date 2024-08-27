import torch
from torch.utils.data import DataLoader
from modlee.timeseries_dataloader import TimeSeriesDataset
import pandas as pd

def makeDataloader():
    data = pd.read_csv("data/HDFCBANK.csv")
    data.drop(
        columns=["Series", "Symbol", "Trades", "Deliverable Volume", "Deliverble"],
        inplace=True,
    )
    encoder_column = data.columns.tolist()
    dataset = TimeSeriesDataset(
        data=data,
        target="Close",
        time_column="Date",
        encoder_column=encoder_column,
        input_seq=2,
        output_seq=1,
    ).to_dataloader(batch_size=1)

    return dataset


# TODO - is this definition deprecated?
# TIMESERIES_MODELS = [
#     pfm.TemporalFusionTransformer(),
#     pfm.DeepAR(),
#     pfm.LSTM(),
#     pfm.GRU(),
# ]
class simpleModel(torch.nn.Module):
    def __init__(self):
        super(simpleModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
        )

    def forward(self, x):
        # Assuming x is a dictionary with a key 'x' that holds the tensor
        # x = x['x']
        # print(x.shape)
        return self.model(x)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, input_channels=10, sequence_length=20):
        self.num_samples = num_samples
        self.input_channels = input_channels
        self.sequence_length = sequence_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random input data
        x = torch.randn(self.input_channels, self.sequence_length)  # For conv1dModel
        # Generate random target data
        y = torch.randn(1)
        return {"x": x}, y


dummy_data = DummyDataset(num_samples=10, input_channels=10, sequence_length=10)


def dummyDataloader(dummy_data1=dummy_data):
    return DataLoader(dummy_data1, batch_size=1, shuffle=False)


# Usage for simpleModel
def get_input_for_simple_model():
    dataloader = dummyDataloader()
    for batch in dataloader:
        x, y = batch
        x = x["x"]
        x = x.reshape(1, -1)
    return x, y


# Usage for conv1dModel
def get_input_for_conv1d_model():
    dataloader = dummyDataloader()
    for batch in dataloader:
        x, y = batch
        x = x["x"]
        x = x.unsqueeze(0)
    return x, y


class simpleLSTM(torch.nn.Module):
    def __init__(self):
        super(simpleLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=10, hidden_size=10)
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class conv1dModel(torch.nn.Module):
    def __init__(self):
        super(conv1dModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3)
        self.conv2 = torch.nn.Conv1d(in_channels=10, out_channels=1, kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        # x = x['x']
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        return x


TIMESERIES_MODELS = [
    # pfm.TemporalFusionTransformer(),
    # pfm.LSTM(input_size=1, hidden_size=10),
    # pfm.GRU(input_size=1, hidden_size=10, num_layers=1),
    # pfm.RecurrentNetwork()
    # simpleLSTM(),
    simpleModel(),
    conv1dModel(),
    # pfm.AutoRegressiveBaseModel(),
    # NbeatsInit(),
]



DATALOADER = timeseries_loaders = [
    get_input_for_simple_model(),
    get_input_for_conv1d_model(),
    makeDataloader(),
]

TIMESERIES_MODALITY_TASK_MODEL = []

TIMESERIES_MODALITY_TASK_KWARGS = [
    ("timeseries", "classification", {"_type": "simple"}),
    ("timeseries", "classification", {"_type": "conv"}),
]


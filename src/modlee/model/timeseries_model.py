import torch
from modlee.model import ModleeModel


class TimeseriesModleeModel(ModleeModel):
    def __init__(self, ):
        pass


class TimeseriesClassificationModleeModel(ModleeModel):
    def  __init__(self, _type="simple"):
        super().__init__()
        if _type=="conv":
            self.model = conv1dModel()
        else:
            self.model = simpleModel()
            
    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)


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

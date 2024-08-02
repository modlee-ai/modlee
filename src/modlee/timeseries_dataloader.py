import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

class TimeSeriesDataset(Dataset):
    """
    Class to handle data loading of the time series dataset.

    """
    def __init__(self, data, target, input_seq:int, output_seq:int):
        """
        Params:
        -------
        data: pands.DataFrame
            The data to be used for training.
        target: str
            The target column name.
        input_seq: int
            The number of input sequence.
        output_seq: int
            The number of output sequence.
        """
        
        self.data = data
        self.target = target
        self.input_seq = input_seq
        self.output_seq = output_seq if output_seq > 0 else 1

    def __len__(self):
        return len(self.data) - self.input_seq - self.output_seq + 1
    
    def __getitem__(self, idx):
        """
        Params:
        -------
        idx: int
            The index of the data to be loaded.
        """
        
        idx = idx + self.input_seq
        x = self.data.iloc[idx - self.input_seq:idx].values
        y = self.data.iloc[idx:idx + self.output_seq][self.target].values
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
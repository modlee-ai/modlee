from torch.utils.data import Dataset, DataLoader
import pytorch_forecasting as pf
import torch
import pandas as pd


class TimeSeriesDataset(Dataset):
    """
    Class to handle data loading of the time series dataset.
    """
    def __init__(self, data, target, input_seq: int, output_seq: int, time_column: str, encoder_column: list):
        """
        Params:
        -------
        data: pandas.DataFrame
            The data to be used for training.
        target: str
            The target column name.
        input_seq: int
            The number of input sequences.
        output_seq: int
            The number of output sequences.
        time_column: str
            The name of the time column.
        """
        self.data = data
        self.target = target
        self.time_column = time_column
        self.encoder_columns = encoder_column

        # Convert time column to datetime if necessary
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.time_column]):
            try:
                self.data[self.time_column] = pd.to_datetime(self.data[self.time_column])
            except Exception as e:
                raise ValueError(f"Could not convert {self.time_column} to datetime. {e}")

        self.data['time_idx'] = (self.data[self.time_column] - self.data[self.time_column].min()).dt.days

        self.input_seq = input_seq
        self.output_seq = output_seq if output_seq > 0 else 1

        self.dataset = pf.TimeSeriesDataSet(
            self.data,
            time_idx='time_idx',
            target=self.target,
            group_ids=[self.target],
            min_encoder_length=self.input_seq,
            max_encoder_length=self.input_seq,
            min_prediction_length=self.output_seq,
            max_prediction_length=self.output_seq,
            time_varying_unknown_reals=[self.target],
            allow_missing_timesteps=True
        )

        self._length = len(self.data) - self.input_seq - self.output_seq + 1
        if self._length <= 0:
            raise ValueError("The dataset length is non-positive. Check the input and output sequence lengths.")
        print(f"Dataset length: {self._length}")

    def to_dataloader(self, batch_size: int=32, shuffle: bool = False):
        return self.dataset.to_dataloader(batch_size=batch_size, shuffle=shuffle)
        
    

class TimeSeriesDataset(Dataset):
    """
    Class to handle data loading of the time series dataset.
    """
    def __init__(self, data, target, input_seq:int, output_seq:int, time_column:str):
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
        self.time_column = time_column
        self.encoder_columns = encoder_column

        if not pd.api.types.is_datetime64_any_dtype(self.data[self.time_column]):
            try:
                self.data[self.time_column] = pd.to_datetime(self.data[self.time_column])
            except Exception as e:
                raise ValueError(f"Could not convert {self.time_column} to datetime. {e}")

        self.data['time_idx'] = (self.data[self.time_column] - self.data[self.time_column].min()).dt.days

        self.input_seq = input_seq
        self.output_seq = output_seq if output_seq > 0 else 1
        self.time_column = time_column
        # Convert time column to datetime if necessary
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.time_column]):
            try:
                self.data[self.time_column] = pd.to_datetime(self.data[self.time_column])
            except Exception as e:
                raise ValueError(f"Could not convert {self.time_column} to datetime. {e}")

        self.data[time_column] = (self.data[self.time_column] - self.data[self.time_column].min()).dt.days

        self.dataset = pf.TimeSeriesDataSet(
            self.data,
            time_idx='time_idx',
            target=self.target,
            group_ids=[self.target],
            min_encoder_length=self.input_seq,
            max_encoder_length=self.input_seq,
            min_prediction_length=self.output_seq,
            max_prediction_length=self.output_seq,
            time_varying_unknown_reals=[self.target],
            allow_missing_timesteps=True
        )

        self._length = len(self.data) - self.input_seq - self.output_seq + 1
        if self._length <= 0:
            raise ValueError("The dataset length is non-positive. Check the input and output sequence lengths.")
        print(f"Dataset length: {self._length}")

        self.dataset = pf.TimeSeriesDataSet(
            self.data,
            time_idx='time_idx',
            target=self.target,
            group_ids=[self.target],
            min_encoder_length=self.input_seq,
            max_encoder_length=self.input_seq,
            min_prediction_length=self.output_seq,
            max_prediction_length=self.output_seq,
            time_varying_unknown_reals=[self.target],
            allow_missing_timesteps=True
        )

        self._length = len(self.data) - self.input_seq - self.output_seq + 1
        if self._length <= 0:
            raise ValueError("The dataset length is non-positive. Check the input and output sequence lengths.")
        print(f"Dataset length: {self._length}")

    def __len__(self):
        return self._length
    
    def get_dataset(self):
        return self.dataset

    def to_dataloader(self, batch_size: int=32, shuffle: bool = False):
        return self.dataset.to_dataloader(batch_size=batch_size, shuffle=shuffle)
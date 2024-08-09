import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from modlee.data_metafeatures import TimeSeriesDataMetafeatures
from modlee.timeseries_dataloader import TimeSeriesDataset

data = pd.read_csv('data/HDFCBANK.csv', parse_dates=['Date'])
data.drop(columns=['Symbol', 'Series', 'Trades', 'Deliverable Volume', 'Deliverble'], inplace=True)
print(data.info())
'''print("Data shape: {}".format(data.shape))
## we normalize the data
data['Close'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()
data['Volume'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()
data['Open'] = (data['Open'] - data['Open'].mean()) / data['Open'].std()
data['High'] = (data['High'] - data['High'].mean()) / data['High'].std()
data['Low'] = (data['Low'] - data['Low'].mean()) / data['Low'].std()
data['Adjusted Close'] = (data['Adjusted Close'] - data['Adjusted Close'].mean()) / data['Adjusted Close'].std()'''



encoder_columns = data.columns.tolist()
print(f"Encoder columns: {encoder_columns}")
dataset = TimeSeriesDataset(data=data, target='Close', input_seq=5, output_seq=2, time_column='Date', encoder_column = encoder_columns)
#print(f"Number of series/groups after filtering: {len(dataset.dataset)}")
#dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
dataloader = dataset.to_dataloader(batch_size=32, shuffle=False)
print(f"Number of batches: {len(dataloader)}")
print(f"Batch size: {dataloader.batch_size}")
##print data shape after loading into dataloader

# Inspect a single batch
for i, batch in enumerate(dataloader):
    print(f"Processing batch {i}")
    if isinstance(batch, dict):
        print(f"Batch keys: {batch.keys()}")
        if 'encoder_cont' in batch and 'decoder_target' in batch:
            features = batch['encoder_cont']
            targets = batch['decoder_target']
            print(f"Features shape: {features.shape}")
            print(f"Targets shape: {targets.shape}")
        else:
            print(f"Expected keys not found in batch {i}")
    else:
        print(f"Batch {i} is not in expected format")
    break


meta = TimeSeriesDataMetafeatures(dataloader)
if not meta:
    raise ValueError("No metafeatures calculated. Please check the input data and parameters.")

metafeatures = meta.calculate_metafeatures()

# Calculate the maximum length of the keys
meta.print_meta(metafeatures)
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from modlee.data_metafeatures import TimeSeriesDataMetafeatures
from modlee.timeseries_dataloader import TimeSeriesDataset, TimeSeriesDataset1

data = pd.read_csv('data/ACGBY.csv')
#data.drop(columns=['Symbol', 'Series', 'Trades', 'Deliverable Volume', 'Deliverble'], inplace=True)
print(data.info())

dataset = TimeSeriesDataset1(data=data, target='Close', input_seq=1, output_seq=1, time_column='Date')
#print(f"Number of series/groups after filtering: {len(dataset.dataset)}")
#dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
dataloader = dataset.to_dataloader(batch_size=32, shuffle=False)

for i, batch in enumerate(dataloader):
    #print(f"Processing batch {i}")
    try:
        if batch is None:
            raise ValueError(f"Batch {i} is None.")
        #print(f"Batch {i} content: {batch}")
    except Exception as e:
        print(f"Error processing batch {i}: {e}")

meta = TimeSeriesDataMetafeatures(dataloader)
if not meta:
    raise ValueError("No metafeatures calculated. Please check the input data and parameters.")

metafeatures = meta.calculate_metafeatures()

# Calculate the maximum length of the keys
max_key_length = max(len(key) for key in metafeatures.keys())

# Print each key-value pair with the keys aligned
for key, value in metafeatures.items():
    print(f"{key:<{max_key_length}} : {value}")
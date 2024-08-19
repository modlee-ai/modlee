#%%
import torch
import torch.onnx
from pytorch_forecasting import models as pfm
from pytorch_forecasting import NBeats, AutoRegressiveBaseModel, DecoderMLP
import pandas as pd
from lightning.pytorch import Trainer

# Load data
data = pd.read_csv('data/HDFCBANK.csv')
data.drop(columns=['Series', 'Symbol','Trades', 'Deliverable Volume', 'Deliverble'], inplace=True)
encoder_column = data.columns.tolist()
#%%
print(data.describe())
print(data.isnull().sum())
#%%
from modlee.timeseries_dataloader import TimeSeriesDataset
import torch.onnx

dataset = TimeSeriesDataset(data=data, encoder_column=encoder_column, target = 'Close', time_column = 'Date', input_seq=2, output_seq=1)
trainer = Trainer(
    max_epochs=3,
    accelerator="auto",
    enable_model_summary=True,
    gradient_clip_val=0.01,
    limit_train_batches=150,
)
model = NBeats.from_dataset(
    dataset=dataset.get_dataset(), 
)
from modlee.data_metafeatures import TimeSeriesDataMetafeatures
print(dataset.to_dataloader())
meta = TimeSeriesDataMetafeatures(dataset.to_dataloader())
features = meta.calculate_metafeatures()
print(features)
#meta.print_meta(features=features)

#trainer.fit(model=model, train_dataloaders=dataset.to_dataloader())

model.eval()
sample_data = next(iter(dataset.to_dataloader(batch_size=1)))

print(sample_data)
print(type(sample_data))

x = {'x':sample_data[0]}
#%%
from modlee.converter import Converter

converter = Converter()

onnx_model = converter.torch_model2onnx_graph(model, input_dummy=x)

print("Onnx Conversion complete >>> Graph below: ")
print(onnx_model)

input_names = ["encoder_cont"]
output_names = ["output"]
dynamic_axes = {"encoder_cont": {0: "batch_size", 1: "time_steps"}, "output": {0: "batch_size"}}
#%%
# Export the model to ONNX format
torch.onnx.export(
    model,
    (x),
    "nbeats_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes
)
print("Model converted to ONNX format successfully")
# %%

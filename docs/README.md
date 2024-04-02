# modlee

## Introduction
Modlee is a machine learning tool that **documents** experiments for reproduciblity and **recommends** neural network models suited for a particular dataset.
Modlee bypasses costly machine learning experimentation by recommending performant models based on prior experiments.
Modlee documents traditional experiment assets (model checkpoints, (hyper)parameters, performance metrics) and meta-features for [meta-learning](https://ieeexplore.ieee.org/abstract/document/9428530).
Based on these meta-features from prior experiments, Modlee recommends a neural network model matched to a new task.

## Installation
The Modlee package consists of the client-side interface for the recommendation API and auxiliary functions for documentation.
The documentation functionality is usable without an account, but the recommendation functionality requires an API key.
To create an account and receive an API key, [visit our website](https://www.dashboard.modlee.ai).

### PyPI
Install `modlee` from PyPI:
```shell
pip install modlee
```

### Source
Alternatively, build the package from the source by cloning this repository and installing from the `pyproject.toml` configuration file:
```shell
git clone https://github.com/modlee-ai/modlee
cd modlee
pip install .
```

We have developed the package in Python 3.10. 
Please [raise an issue](https://github.com/modlee-ai/modlee/blob/main/issues)) if you experience environment errors.

### Set API key
Either save your API key to an environment variable: 
```shell
export MODLEE_API_KEY="my-api-key"
```
Or pass directly to the  `modlee.init` function (less recommended):
```python
# your_experiment_script.py
import modlee
modlee.init(api_key="my-api-key")
```

## Usage

Modlee is built on top of [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [MLFlow](https://mlflow.org).
While you do not have to be an expert in either framework to use Modlee, we recommend having at least a familiarity with machine learning and the experiment pipeline.
This documentation page does not cover the frameworks; we recommend referencing the [Lightning](https://lightning.ai/docs/overview/getting-started) and [MLFlow](https://mlflow.org/docs/latest/index.html) documentation directly.

### Document
Modlee supports documentation for Lightning experiments.
Guides for structuring PyTorch Lightning projects are available [here](https://lightning.ai/docs/pytorch/stable/starter/converting.html) and [here](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09).
Once you have created your experiment script, simply follow the four "I's":
```python
# Import the package
import modlee, lightning

# Initialize with your API key
modlee.init(api_key="my-api-key")

# Inherit the ModleeModel class for your model module,
# instead of lightning.pytorch.LightningModule
class MyModel(modlee.model.ModleeModel):
    # Define the model
model = MyModel()

# Insert the modlee context manager before training
with modlee.start_run() as run:
    trainer = modlee.trainer(max_epochs=10)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
    )
```

Modlee will document experiment assets in a new `./mlruns/` directory, relative to wherever the script was called.
Among the assets is a `model_graph.py` module that recreates the model as a graph, including the `forward()` pass:

```python
import torch, onnx2torch
from torch import tensor

class Model(torch.nn.Module):
    
    def __init__(self):
        ''' Rebuild the model from its base components. '''
        super().__init__()
        setattr(self,'Shape', Shape(**{'start':0,'end':None}))
        setattr(self,'Constant', Constant(**{'value':torch.ones(())*2}))
        setattr(self,'Gather', Gather(**{'axis':0}))
        setattr(self,'Shape_1', Shape(**{'start':0,'end':None}))
        setattr(self,'Constant_1', Constant(**{'value':torch.ones(())*3}))
        setattr(self,'Gather_1', Gather(**{'axis':0}))
        setattr(self,'Conv', torch.nn.modules.conv.Conv2d(**{
            'in_channels':3,
            'out_channels':64,
            'kernel_size':(7, 7),
            'stride':(2, 2),
            'padding':(3, 3),
            'dilation':(1, 1),
            'groups':1,
            'padding_mode':'zeros'}))
        ...
    
    def forward(self, input_1):
        ''' Forward pass an input through the network '''
        shape = self.Shape(input_1)
        constant = self.Constant()
        gather = self.Gather(shape, constant.type(torch.int64))
        shape_1 = self.Shape_1(input_1)
        constant_1 = self.Constant_1()
        gather_1 = self.Gather_1(shape_1, constant_1.type(torch.int64))
        conv = self.Conv(input_1)
        ...

```

### Recommend
Modlee recommends models based on your data modality, task, and data meta-features.
Rather than defining the model manually, you can use this recommended model as a starting point for your experiments.
```python
# Import and initialize
import modlee, lightning
modlee.init(api_key="my-api-key")

# Create your dataloaders
train_dataloader, val_dataloader = your_function_to_get_dataloaders()

# Create a recommender object and fit to the training dataloader
recommender = modlee.recommender.from_modality_task(
    modality='image',
    task='classification',
    )

# Fit the recommender to the data meta-features
recommender.fit(train_dataloader)

# Get the model from the recommender and train
model = recommender.model
with modlee.start_run() as run:
    trainer = modlee.Trainer(max_epochs=10)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
```

## Support

### Contributing
We welcome contributions of any kind: bug reports, feature requests, tutorials, etc.
Before submitting a pull request, [please read the contribution guidelines](https://github.com/modlee-ai/modlee/blob/main/docs/CONTRIBUTING.md).

### Issues
If you encounter errors, [please raise an issue in this repository](https://github.com/modlee-ai/modlee/issues).

### Community
[Join our Discord server](https://discord.com/invite/m8YDbWDvrF) to discuss and contribute with other Modlee users.

## Roadmap
- [ ] Add more modalities and tasks.
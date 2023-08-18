# modlee_pypi

modlee helps you document your machine learning experiments.
Built over the widely adopted [`mlflow` platform](https://mlflow.org), modlee logs parameters and performance in the correct format for training the community- suggestion model.

## Structure
We currently support [Lightning](https://github.com/Lightning-AI/lightning) PyTorch models.
Lightning wraps over PyTorch modules and simplifies the development cycle by organizing models, (hyper)parameters, datasets, and training loops in a single class.
`lightning.pytorch.LightningModules` handle the boilerplate while preserving the experimentation underneath at the `torch.nn.Module` level.
Some guides for migrating PyTorch code into Lightning are [here](https://lightning.ai/docs/pytorch/stable/starter/converting.html) and [here](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09).

*Note: the `lightning` packaged evolved from the similarly named `pytorch-lightning` package, which is now deprecated. If there are any compatibility issues, remove/uninstall with `python3 -m pip uninstall pytorch-lightning`.*

## Usage
Currently working on python3.10.
### Installation
First, clone and enter this repository:
```
git clone https://github.com/modlee-ai/modlee_pypi.git
cd modlee_pypi
```

Make a virtual environment using python >= 3.10 and install this repository as a package:
```
python3.10 -m venv venv
source venv/bin/activate
python3 -m pip install .
```
Download a specific version of python >= 3.10 [here](https://www.python.org/downloads/)

## Implementing 

### Model definition
Given a `lightning.pytorch.LightningModule`:
```
import lightning

class ExampleModel(lightning.pytorch.LightningModule):
...
```

Change the subclass to `modlee_pypi.modlee_model.ModleeModel`:
```
import modlee_pypi

class ExampleModel(modlee_pypi.modlee_model.ModleeModel):
...
```
### Documenting
Before training, wrap the call to the `lightning.pytorch.Trainer` with a call to `modlee_pypi.start_run()`.
This will begin logging:
```
with modlee_pypi.start_run() as run:
    trainer = lightning.pytorch.Trainer()
    trainer.fit(
        model=model,
        train_dataloaders=training_loader,
        val_dataloaders=test_loader)
```

The logs (artifacts, parameters, metrics, etc) will be saved by default to `./mlruns`.

### Accessing old experiments

### Sharing experiments

## Examples
Refer to [`./notebooks`](./notebooks) for examples, executable as either plain Python scripts or as Jupyter-*like* notebooks in VS code.

## Implementation notes
modlee saves a snapshot of the model source.
To ensure that this snapshot is valid, any necessary functions or variables should be defined as attributes of your model or subclasses of `torch.nn.Module`.

For example, instead of defining `batch_size` and `build_model` outside of `ExampleModel`:
```
batch_size = 64
def build_model():
    # model building goes here
    return model

class ExampleModel(modlee_pypi.modlee_model.ModleeModel):
    def __init__(self, *args, **kwargs):
        self.batch_size = batch_size
        model = build_model()
```

Define `batch_size` inside and make `build_model` a subclass of `torch.nn.Module`:
```
class build_model(torch.nn.Module):
    def __init__(self,):
        # model building goes here
        

class ExampleModel(modlee_pypi.modlee_model.ModleeModel):
    def __init__(self, *args, **kwargs):
        self.batch_size = 64
        model = build_model()
```

## Troubleshooting

### GPU issues on Apple Silicon
Install PyTorch nightly:
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### pytorch vision pretrained model weight download SSL issue
```
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

## TODO
- [ ] Access prior models - (re)load given an experiment / run ID, 
  - [ ] Maybe we could use a compilation check (e.g. try to call ModleeModel()) to make sure that the model builds. If it fails, show an error or warning to the user to indicate that this model is not properly documented.
    - [ ] Any way to check what variables are defined?


## Brainstorming TODO
- [ ] make notebooks toggle-able for local files and package (toggle where we import mlflow from)
- [ ] can we incorporate the model code_to_text within a testing notebook?
- [ ] Right now it seems like we are just saving mlruns in the same folder as the python script right? Eventually, we may want to have modlee automatically keep track of where these are if we plan 

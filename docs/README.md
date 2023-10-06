# modlee

modlee documents machine learning experiments.
modlee logs assets (e.g. parameters, performance metrics, data complexity) in a proper format for training the community-built suggestion model.

# Structure
We currently support [Lightning](https://github.com/Lightning-AI/lightning) PyTorch models.
Lightning wraps over PyTorch modules and simplifies the development cycle by organizing models, (hyper)parameters, datasets, and training loops in a single class.
`lightning.pytorch.LightningModules` handle the boilerplate while preserving the experimentation underneath at the `torch.nn.Module` level.
Some guides for migrating PyTorch code into Lightning are [here](https://lightning.ai/docs/pytorch/stable/starter/converting.html) and [here](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09).

**Note:** the `lightning` packaged evolved from the similarly named `pytorch-lightning` package, which is now deprecated but still used in some backend mechanisms of `mlflow`. To avoid compatibility issues, ensure that you are using objects from `lightning.torch` instead of `pytorch_lightning`:
```
import pytorch_lightning as pl # DO NOT USE THIS
import lightning.pytorch as pl # use this instead
```

# Usage
Currently working on python3.10.
## Installation
First, clone and enter this repository:
```
git clone https://github.com/modlee-ai/modlee_pypi.git
cd modlee_pypi
```

Make a virtual environment using python >= 3.10 (available [here](https://www.python.org/downloads/)) and install this repository as a package:
```
python3.10 -m venv venv
source venv/bin/activate
python3 -m pip install .
```

## Implementing 
This section details the steps to use `modlee` to log experiments.
A minimal working example is at `examples/simple_torch.py`.


### Import and initialize
At the head of the script that runs the training loop (i.e. wherever you call `lightning.pytorch.Trainer.fit()`), import the `modlee` package and initialize with your API key.
```
import lightning
import lightning.pytorch as pl

import modlee
modlee.init(api_key='my_api_key')
```
By default, `modlee.init()` will log experiments to a `./mlruns/` folder in the same directory as the script.
You can define a different directory with `modlee.init(run_dir='path/to/save/experiments',api_key='my_api_key')`.
The `run_dir` path will be interpreted relative to the current script.

By default, the package will make requests to the remote server endpoint at `http://modlee.pythonanywhere.com`.
If you have the local `modlee_api` server running, set `api_key='local'` to route requests to the local endpoint at `http://127.0.0.1:7070`.

### Model definition
Converting a base Lightning model to a Modlee model with built-in logging requires simply changing its parent class.
Given a `lightning.pytorch.LightningModule`:
```
class ExampleModel(lightning.pytorch.LightningModule):
...
```

Change the parent class to inherit from `modlee.modlee_model.ModleeModel`:
```
class ExampleModel(modlee.modlee_model.ModleeModel):
...
```

Requirements for structuring the model to ensure reproducibility are [below](#implementation-notes).

### Documenting
To log during training, wrap the call to the `lightning.pytorch.Trainer` with a call to `modlee.start_run()`.
```
with modlee.start_run() as run:
    trainer = lightning.pytorch.Trainer()
    trainer.fit(
        model=model,
        train_dataloaders=training_loader,
        val_dataloaders=test_loader)
```

Modlee will save the assets (e.g. data statistics, model text representation) to a new run directory for each experiment.

### Retrieving old experiments
Functions for retrieving assets from old experiments are in `modlee/retriver.py`.
You can retrieve the model and data snapshots.

### Sharing experiments


## Examples
Refer to [`./examples`](./examples) for examples, executable as either plain Python scripts or as Jupyter-*like* notebooks in VS code.

## Implementation notes
modlee saves a snapshot of the model source.
To ensure that this snapshot is valid, any necessary functions or variables should be defined as either 1) internal attributes of your model or 2) subclasses of `torch.nn.Module`.

For example, instead of defining `batch_size` and `build_model` outside of `ExampleModel`:
```
batch_size = 64
def build_model():
    # model building goes here
    return model

class ExampleModel(modlee.modlee_model.ModleeModel):
    def __init__(self, *args, **kwargs):
        self.batch_size = batch_size
        model = build_model()
```

Define `batch_size` inside and make `build_model` a subclass of `torch.nn.Module`:
```
class build_model(torch.nn.Module):
    def __init__(self,):
        # model building goes here
        

class ExampleModel(modlee.modlee_model.ModleeModel):
    def __init__(self, *args, **kwargs):
        self.batch_size = 64
        model = build_model()
```

We are currently experimenting with how automatically logging custom parameters as assets, e.g. how `model = ExampleModel(custom_parameter=custom_value)` could log `custom_parameter:custom_value` as any other asset.
This is partially working by setting parameters as object attributes.
For example:
```
class ExampleModel(modlee.modlee_model.ModleeModel):
    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
```
This will log the variable as an artifact at the beginning of training and enable reconstructing the object later.
The variable must be JSON serializable, i.e. a simple type like `int` or `str`; functions cannot be serialized.

# Creating package
## Build
```
python3 -m build
```
## Upload 
```
# For the testpypi repository
python3 -m twine upload --repository testpypi dist/*

# For the actual repository
python3 -m twine upload dist/*
```

# Troubleshooting

## Unit Tests
We have tests in `tests/`.
To run:
```
cd tests/
python3 -m unittest discover .
```
To run a specific test, replace `discover .` with the script name, e.g. for the API client:
```
python3 -m unittest test_api_client
```

*Note: `test_retriever.py` expects local paths to completed experiments and `mlruns` folders. Modify `mlruns_dirs`, `run_dirs`, and `fail_run_dirs` accordingly.*

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

### Multiprocessing error
The call to `trainer.fit()` may throw an error related to multiprocessing:
```
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...
```

The currently working solution is to run the training loop directly as a script and wrap the training call in `if __name__=="__main__"`:
```
if __name__=="__main__":
    trainer.fit(...)
```

### Loop
Sometimes the call to start training will loop repeatedly, with a warning:
```
[W ParallelNative.cpp:230] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)
```
This may just be an issue with macOS.
I suspect it's related to parallelizing data across different CPUs.
In some of the `lightning_tutorial` examples:
```
NUM_WORKERS = int(os.cpu_count() / 2) # this equals 4 on the MacBook Air M2
train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
```
Either wait it out or try setting `NUM_WORKERS=1`.

## Brainstorming TODO
- [ ] make notebooks toggle-able for local files and package (toggle where we import mlflow from)
- [ ] can we incorporate the model code_to_text within a testing notebook?
- [ ] Right now it seems like we are just saving mlruns in the same folder as the python script right? Eventually, we may want to have modlee automatically keep track of where these are if we plan 

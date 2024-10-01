""" 
Utility functions.
"""
import os, sys, time, json, pickle, requests, importlib, pathlib
import re
from functools import partial
import json
from urllib.parse import urlparse, unquote
from ast import literal_eval
import pickle
import requests
import math, numbers
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import mlflow

import torch
from torch.utils.data import Dataset, DataLoader
import torchtext
import torchvision
from torchvision import datasets as tv_datasets
from torchvision import transforms
from torch.utils.data import DataLoader
# from modlee.timeseries_dataloader import TimeseriesDataset

import modlee
from modlee.client import ModleeClient


from torch.utils.data import Dataset, DataLoader
try:
    import pytorch_forecasting as pf
except:
    pass
import torch
import pandas as pd


class TimeseriesDataset(Dataset):
    """
    Class to handle data loading of the time series dataset.
    """
    def __init__(self, data, target, input_seq: int, output_seq: int, time_column: str, encoder_column: list):
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

# g2v = ModelEncoder.from()
INPUT_DUMMY = {
    "image": torch.randn([10,3,300,300]),
    "": torch.randn([10,3,300,300]),
    # "tabular": {
    #     "x": torch.randn([10, 3, 300, 300]),
    # },
    "tabular": torch.randn([1,10]),
    "text": [
        "hello world",
        "the quick brown fox jumps over the lazy dog" * 10,
        "the mitochondria is the powerhouse of the cell"
    ],
    "timeseries":  next(iter(dummy_data))[0]
}

def safe_mkdir(target_path):
    """
    Safely make a directory.

    :param target_path: The path to the target directory.
    """
    root, ext = os.path.splitext(target_path)
    # is a file
    if len(ext) > 0:
        target_path = os.path.split(root)
    else:
        target_path = f"{target_path}/"
    # if os.path.isfile(target_dir):
    #     target_dir,_ = os.path.split(target_dir.split('.')[0])
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def get_fashion_mnist(batch_size=64, num_output_channels=1):
    """
    Get the Fashion MNIST dataset from torchvision.

    :param batch_size: The batch size, defaults to 64.
    :param num_output_channels: Passed to torchvision.transforms.Grayscale. 1 = grayscale, 3 = RGB. Defaults to 1.
    :return: A tuple of train and test dataloaders.
    """
    data_transforms = torchvision.transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=num_output_channels),
            transforms.ToTensor(),
        ]
    )
    training_loader = DataLoader(
        tv_datasets.FashionMNIST(
            root=".data", train=True, download=True, transform=data_transforms
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        tv_datasets.FashionMNIST(
            root=".data", train=False, download=True, transform=data_transforms
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return training_loader, test_loader


def get_imagenette_dataloader():
    """
    Get a small validation dataloader for imagenette (https://pytorch.org/vision/stable/generated/torchvision.datasets.Imagenette.html#torchvision.datasets.Imagenette)
    """
    return get_dataloader(
        torchvision.datasets.Imagenette(
            root=".data",
            split="val",
            size="160px",
            download=not os.path.exists(".data/imagenette2-160"),
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((160, 160))]
            ),
        ),
    )


def get_dataloader(dataset, batch_size=16, shuffle=True, *args, **kwargs):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, *args, **kwargs)


class timeseries_loader:
    @staticmethod
    def get_timeseries_dataloader(data, target, input_seq:int, output_seq:int, time_column:str, encoder_column:list):
        return TimeseriesDataset(
                data, target, input_seq, output_seq, time_column, encoder_column, 
            ).to_dataloader(batch_size=32, shuffle = False)

class image_loaders:
    @staticmethod
    def _get_image_dataloader(dataset_module, root=".data", *args, **kwargs):
        kwargs["transform"] = kwargs.get(
            "transform",
            torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((300, 300)),
                    torchvision.transforms.Grayscale(num_output_channels=3),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        # print(args, kwargs)
        return get_dataloader(
            getattr(torchvision.datasets, dataset_module)(
                root=root,
                # split=split,
                # download=True,
                **kwargs,
            )
        )

    image_modules = {
        # 'Caltech101'
        # 'CelebA':{
        #     'split':'test',
        #     'download':True,
        #     },
        "CIFAR10": dict(train=False, download=True),  # Passed
        # 'Country211':dict(
        #     split='test',
        #     download=True),
        "DTD": dict(split="test", download=True),  # Passed
        # 'EMNIST':dict(            # File download issues
        #     split="byclass",
        #     train=False,
        #     download=True),
        "EuroSAT": dict(download=True),  # Passed
        "FashionMNIST": dict(train=False, download=True),  # Passed
        # 'FER2013':dict(             # Cannot download
        #     split='test',),
        "FGVCAircraft": dict(split="test", download=True),  # Passed
        # # 'Flicker8k',
        "Flowers102": dict(split="test", download=True),  # Passed
        # 'Food101':dict(             # Took too long to download
        #     split='test',
        #     download=True),
        "GTSRB": dict(split="test", download=True),  # Passed
        # 'INaturalist':dict(         # Too big — over 8GB for validation
        #     version='2021_valid',
        #     download=True),
        "Imagenette": dict(  # Passed
            split="val",
            size="full",
            # download=True
        ),
        "KMNIST": dict(train=False, download=True),  # Passed
        # 'LFWPeople':dict(         # Corrupt file
        #     split='test',
        #     download=True),
        # 'LSUN':dict(              # Uses deprecated package
        #     classes='test'),
        "MNIST": dict(train=False, download=True),  # Passed
        "Omniglot": dict(download=True),  # Passed
        "OxfordIIITPet": dict(split="test", download=True),  # Passed
        "Places365": dict(  # Passed
            split="val",
            small=True,
            # download=True,
        ),
        # 'PCAM':dict(                # Took too long to download
        #     split='test',
        #     download=True),
        "QMNIST": dict(what="test10k", download=True),  # Passed
        "RenderedSST2": dict(split="test", download=True),  # Passed
        "SEMEION": dict(download=True),  # Passed
        # 'SBU':dict(               # Took too long to download
        #     download=True),
        # 'StanfordCars':dict           # Not available(
        #     split='test',
        #     download=True),
        "STL10": dict(split="test", download=True),  # Passed
        "SUN397": dict(download=True),  # Passed
        "SVHN": dict(split="test", download=True),  # Passed
        "USPS": dict(train=False, download=True),  # Passed
    }
    # take only a few
    ctr = 0
    for image_module, kwargs in image_modules.items():
        locals()[f"get_{image_module.lower()}_dataloader"] = partial(
            _get_image_dataloader, image_module, **kwargs
        )
        ctr += 1
        if ctr == 4:
            break


class text_loaders:
    @staticmethod
    def _get_text_dataloader(dataset_module, dataset_len, root=".data", split="dev"):
        return get_dataloader(
            getattr(torchtext.datasets, dataset_module)(
                root=root, split=split
            ).set_length(dataset_len)
        )

    @staticmethod
    def get_mnli_dataloader(*args, **kwargs):
        kwargs["split"] = "dev_matched"
        return get_dataloader(torchtext.datasets.MNLI(**kwargs).set_length(9815))

    text_modules_lengths = {
        "STSB": 1500,
        "SST2": 872,
        "RTE": 277,
        "QNLI": 5463,
        "CoLA": 527,
        "WNLI": 71,
        # 'SQuAD1': 10570,
        # 'SQuAD2': 11873,
    }
    for dataset_module, dataset_len in text_modules_lengths.items():
        locals()[f"get_{dataset_module.lower()}_dataloader"] = partial(
            _get_text_dataloader, dataset_module, dataset_len
        )

class tabular_loaders:

    class TabularDataset(Dataset):
        def __init__(self, data, target):
            self.data = torch.tensor(data, dtype=torch.float32)
            self.target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.target[idx]

        
    @staticmethod
    def get_diabetes_dataloader(batch_size=4, shuffle=True, root=None):
        if root is None:
            dataset_path = os.path.join(os.path.dirname(__file__), "csv", "diabetes.csv")
        else: 
            dataset_path = os.path.join(root, "diabetes.csv")

        df = pd.read_csv(dataset_path)

        X = df.drop("Outcome", axis=1).values
        y = df["Outcome"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dataset = tabular_loaders.TabularDataset(X_scaled, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader


    @staticmethod
    def get_adult_dataloader(batch_size=4, shuffle=True, root=None):
        if root is None:
            dataset_path = os.path.join(os.path.dirname(__file__), "csv", "adult.csv")
        else: 
            dataset_path = os.path.join(root, "adult.csv")


        df = pd.read_csv(dataset_path)

        df = df.replace(" ?", pd.NA).dropna()

        X = df.drop("income", axis=1)
        y = df["income"]

        X_encoded = pd.get_dummies(X, drop_first=True)
        X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
        X_encoded = X_encoded.fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        y = pd.factorize(y)[0]

        dataset = tabular_loaders.TabularDataset(X_scaled, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader


    @staticmethod
    def get_housing_dataloader(batch_size=4, shuffle=True, root=None):

        column_names = [
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
            "MEDV",
        ]
        if root is None:
            path_name = os.path.join(os.path.dirname(__file__), "csv", "housing.csv")
        else: 
            path_name = os.path.join(root, "housing.csv")


        df = pd.read_csv(path_name, header=None, names=column_names, delim_whitespace=True)

        X = df.drop("MEDV", axis=1).values
        y = df["MEDV"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dataset = tabular_loaders.TabularDataset(X_scaled, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

class timeseries_loaders:
    @staticmethod
    def get_finance_dataloader(root=None,):
        if root is None:
            data_path = "./data"
            # data_path = os.path.join("./j")
        dataframe = pd.read_csv(os.path.join(data_path, "HDFCBANK.csv"))
        ###normalize the dataset
        dataframe.drop(
            columns=["Symbol", "Series", "Trades", "Deliverable Volume", "Deliverble"],
            inplace=True,
        )
        encoder_columns = dataframe.columns.tolist()
        dataloader = timeseries_loader.get_timeseries_dataloader(
            data=dataframe,
            input_seq=2,
            output_seq=1,
            encoder_column=encoder_columns,
            target="Close",
            time_column="Date",
        )
        return dataloader

def uri_to_path(uri):
    """
    Convert a URI to a path.

    :param uri: The URI to convert.
    :return: The converted path.
    """
    parsed_uri = urlparse(uri)
    path = unquote(parsed_uri.path)
    return path


def is_cacheable(x):
    """
    Check if an object is cacheable / serializable.

    :param x: The object to check cacheability, probably a dictionary.
    :return: A boolean of whether the object is cacheable or not.
    """
    try:
        json.dumps(x)
        return True
    except:
        return False


def get_model_size(model, as_MB=True):
    """
    Get the size of a model, as estimated from the number and size of its parameters.

    :param model: The model for which to get the size.
    :param as_MB: Whether to return the size in MB, defaults to True.
    :return: The model size.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_size = param_size + buffer_size
    if as_MB:
        model_size /= 1024**2
    return model_size


def quantize(x):
    """
    Quantize an object.

    :param x: The object to quantize.
    :return: The object, quantized.
    """

    if float(x) < 0.1:
        ind = 2
        while str(x)[ind] == "0":
            ind += 1
        # print(ind)
        c = np.around(float(x), ind - 1)
    elif float(x) < 1.0:
        c = np.around(float(x), 2)
    elif float(x) < 10.0:
        c = int(x)
    else:
        c = int(2 ** np.round(math.log(float(x)) / math.log(2)))

    return c


_discretize = quantize


def convert_to_scientific(x):
    """
    Convert a number to scientific notation.

    :param x: The number to convert.
    :return: The number in scientific notation as a string.
    """
    return f"{float(x):0.0e}"


def closest_power_of_2(x):
    """
    Round a number to its closest power of 2, i.e. y = 2**floor(log_2(x)).

    :param x: The number.
    :return: The closest power of 2 of the number.
    """
    # Handle negative numbers by taking the absolute value
    x = abs(x)

    # Find the exponent (log base 2)
    exponent = math.log2(x)

    # Round the exponent to the nearest integer
    rounded_exponent = round(exponent)

    # Calculate the closest power of 2
    closest_value = 2**rounded_exponent

    return closest_value


def _is_number(x):
    """
    Check if an object is a number.

    :param x: The object to check.
    :return: Whether the object is a number.
    """
    # if isinstance(n,list):
    #     return all([_is_number(num) for num in n])
    try:
        float(x)  # Type-casting the string to `float`.
        # If string is not a valid `float`,
        # it'll raise `ValueError` exception
    # except ValueError, TypeError:
    except:
        return False
    return True


def quantize_dict(base_dict, quantize_fn=quantize):
    """
    Quantize a dictionary.

    :param base_dict: The dictionary to quantize.
    :param quantize_fn: The function to use for quantization, defaults to quantize.
    :return: The quantized dictionary.
    """
    for k, v in base_dict.items():
        if isinstance(v, dict):
            base_dict.update({k: quantize_dict(v, quantize_fn)})
        elif isinstance(v, (int, float)):
            base_dict.update({k: quantize_fn(v)})
        elif _is_number(v):
            base_dict.update({k: quantize_fn(float(v))})

        # elif 'float' in str(type(v)):
        #     base_dict.update({k:str(v)})
        # elif isinstance(v,np.int64):
        #     base_dict.update({k:int(v)})
    return base_dict


def typewriter_print(text, sleep_time=0.001, max_line_length=150, max_lines=20):
    """
    Print a string letter-by-letter, like a typewriter.

    :param text: The text to print.
    :param sleep_time: The time to sleep between letters, defaults to 0.001.
    :param max_line_length: The maximum line length to truncate to, defaults to 150.
    :param max_lines: The maximum number of lines to print, defaults to 20.
    """
    if not isinstance(text, str):
        text = str(text)
    text_lines = text.split("\n")

    if len(text_lines) > max_lines:
        text_lines = text_lines[:max_lines] + ["...\n"]

    def shorten_if_needed(line, max_line_length):
        if len(line) > max_line_length:
            return line[:max_line_length] + " ...\n"
        else:
            return line + "\n"

    text_lines = [shorten_if_needed(l, max_line_length) for l in text_lines]

    for line in text_lines:
        for c in line:
            print(c, end="")
            sys.stdout.flush()
            time.sleep(sleep_time)


# ---------------------------------------------


def discretize(n: list[float, int]) -> list[float, int]:
    """
    Discretize a list of inputs

    :param n: The list of inputs to discretize.
    :return: The list of discretized inputs.
    """

    try:
        if type(n) == str:
            n = literal_eval(n)

        if type(n) == list:
            c = [_discretize(_n) for _n in n]
        elif type(n) == tuple:
            n = list(n)
            c = tuple([_discretize(_n) for _n in n])
        else:
            c = _discretize(n)
    except:
        c = n

    return c


def apply_discretize_to_summary(text, info):
    """
    Discretize a summary.

    :param text: The text to discretize.
    :param info: An object that contains different separators.
    :return: The discretized summary.
    """

    # text_split = [ [ p.split(key_val_seperator) for p in l.split(parameter_seperator)] for l in text.split(layer_seperator)]
    # print(text_split)

    text_split = [
        [
            [str(discretize(pp)) for pp in p.split(info.key_val_seperator)]
            for p in l.split(info.parameter_seperator)
        ]
        for l in text.split(info.layer_seperator)
    ]
    # print(text_split)

    text_join = info.layer_seperator.join(
        [
            info.parameter_seperator.join([info.key_val_seperator.join(p) for p in l])
            for l in text_split
        ]
    )
    # print(text_join)

    return text_join


def save_run(*args, **kwargs):
    """
    Save the current run.

    :param modlee_client: The client object that is tracking the current run.
    """
    api_key = os.environ.get("MODLEE_API_KEY")
    ModleeClient(api_key=api_key).post_run(*args, **kwargs)


def save_run_as_json(*args, **kwargs):
    """
    Save the current run as a JSON.

    :param modlee_client: The client object that is tracking the current run.
    """
    api_key = os.environ.get("MODLEE_API_KEY")
    ModleeClient(api_key=api_key).post_run_as_json(*args, **kwargs)


def last_run_path(*args, **kwargs):
    """
    Return the path to the last / most recent run path

    :return: The path to the last run.
    """
    artifact_uri = mlflow.last_active_run().info.artifact_uri
    artifact_path = urlparse(artifact_uri).path
    return os.path.dirname(artifact_path)


# def _f32_to_f16(self,base_dict):
def _make_serializable(base_dict):
    """
    Make a dictionary serializable (e.g. by pickle or json) by converting floats to strings.

    :param base_dict: The dictionary to convert.
    :return: The serializable dict.
    """
    for k, v in base_dict.items():
        if isinstance(v, dict):
            base_dict.update({k: _make_serializable(v)})
        elif "float" in str(type(v)):
            base_dict.update({k: str(v)})
        elif isinstance(v, np.int64):
            base_dict.update({k: int(v)})
    return base_dict


def class_from_modality_task(modality, task, _class, *args, **kwargs):
    """
    Return a Recommender object based on the modality and task.
    Currently supports:

    - image
    --- classification
    --- segmentation
    - text
    --- classification

    :param _class: The class to return, either "Model" or "Recommender"
    :param modality: The modality as a string, e.g. "image", "text".
    :param task: The task as a string, e.g. "classification", "segmentation".
    :return: The RecommenderObject, if it exists.
    """
    submodule = getattr(modlee, _class.lower())
    _class = _class.replace("_","")
    if _class.lower() == "model":
        _class = "ModleeModel"
    elif "Metafeatures" not in _class:
        _class = _class.capitalize()
    ClassObject = getattr(
        submodule, f"{modality.capitalize()}{task.capitalize()}{_class}", None
    )

    assert ClassObject is not None, f"No {_class} for {modality} {task}"
    return ClassObject(*args, **kwargs)


def get_modality_task(obj):
    """
    Get the modality and task of a given object,
    e.g. "image" and "classification" from an ImageClassificationModleeModel

    :param obj: The item to parse.
    """
    potential_classes = ["Recommender", "ModleeModel"]
    # obj_name = ""
    # obj_name = type(obj).__name__
    obj_name_q = [obj]
    MODALITIES = ["image", "text", "tabular", "timeseries"]
    TASKS = ["classification", "regression", "segmentation"]
        
    while len(obj_name_q):
        # obj_name = type(obj_name_q.pop(0)).__name__

        _obj = obj_name_q.pop(0)
        # print(_obj)
        _obj = type(_obj) if not isinstance(_obj, type) else _obj
        # _obj_name = type(_obj).__name__ if not isinstance(_obj, type) else _obj.__name__
        # breakpoint()
        _obj_name = _obj.__name__
        if any([potential_class in _obj_name for potential_class in potential_classes]):
            
            obj_name = _obj.__name__.replace("Recommender","").replace("ModleeModel","")

            ret = re.match(r"([A-Z]\w+)([A-Z]\w*)", obj_name)
            if ret is not None:
                ret = [r.lower() for r in ret.groups()]
                if ret[0] in MODALITIES and ret[1] in TASKS:
                    # breakpoint()
                    break
            # break
        # else:
            # obj_name_q.extend(type(_obj).__bases__)
        obj_name_q.extend(_obj.__bases__)

    # breakpoint()
    # obj_name = type(_obj).__name__.replace("Recommender","").replace("ModleeModel","")
    obj_name = _obj.__name__.replace("Recommender","").replace("ModleeModel","")
    # breakpoint()
    ret = re.match(r"([A-Z]\w+)([A-Z]\w*)", obj_name)
    if ret:
        return (r.lower() for r in ret.groups())
    else:
        # TODO handle modality-task-less models
        return "", ""
        # return None
""" 
Utility functions.
"""
import os, sys, time, json, pickle, requests, importlib, pathlib
import json
from urllib.parse import urlparse, unquote
from ast import literal_eval
import pickle
import requests
import math, numbers
import numpy as np
import mlflow
import torch
import torchvision
from torchvision import datasets as tv_datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from modlee.client import ModleeClient

def safe_mkdir(target_path):
    """
    Safely make a directory.

    :param target_path: The path to the target directory.
    """
    root, ext = os.path.splitext(target_path)
    if len(ext) > 0:
        target_path = os.path.split(root)
    else:
        target_path = f"{target_path}/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)

def get_fashion_mnist(batch_size=64, num_output_channels=1):
    """
    Get the Fashion MNIST dataset from torchvision.

    :param batch_size: The batch size, defaults to 64.
    :param num_output_channels: Passed to torchvision.transforms.Grayscale. 1 = grayscale, 3 = RGB. Defaults to 1.
    :return: A tuple of train and test dataloaders.
    """
    data_transforms = torchvision.transforms.Compose([
        transforms.Grayscale(num_output_channels=num_output_channels),
        transforms.ToTensor(),
    ])
    training_loader = DataLoader(
        tv_datasets.FashionMNIST(
            root="data", train=True, download=True, transform=data_transforms
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        tv_datasets.FashionMNIST(
            root="data", train=False, download=True, transform=data_transforms
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return training_loader, test_loader

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
        model_size /= 1024 ** 2
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
    closest_value = 2 ** rounded_exponent
    return closest_value

def _is_number(x):
    """
    Check if an object is a number.

    :param x: The object to check.
    :return: Whether the object is a number.
    """
    try:
        float(x)  # Type-casting the string to `float`.
        # If string is not a valid `float`,
        # it'll raise `ValueError` exception
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
    text_split = [
        [
            [str(discretize(pp)) for pp in p.split(info.key_val_seperator)]
            for p in l.split(info.parameter_seperator)
        ]
        for l in text.split(info.layer_seperator)
    ]

    text_join = info.layer_seperator.join(
        [
            info.parameter_seperator.join([info.key_val_seperator.join(p) for p in l])
            for l in text_split
        ]
    )
    return text_join

def save_run(*args, **kwargs):
    """
    Save the current run.

    :param modlee_client: The client object that is tracking the current run.
    """
    api_key = os.environ.get('MODLEE_API_KEY')
    ModleeClient(api_key=api_key).post_run(*args, **kwargs)

def save_run_as_json(*args, **kwargs):
    """
    Save the current run as a JSON.

    :param modlee_client: The client object that is tracking the current run.
    """
    api_key = os.environ.get('MODLEE_API_KEY')
    ModleeClient(api_key=api_key).post_run_as_json(*args, **kwargs)

def last_run_path(*args, **kwargs):
    """ 
    Return the path to the last / most recent run path
    
    :return: The path to the last run.
    """
    artifact_uri = mlflow.last_active_run().info.artifact_uri
    artifact_path = urlparse(artifact_uri).path
    return os.path.dirname(artifact_path)
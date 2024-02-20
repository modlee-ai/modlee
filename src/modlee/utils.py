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

from torchvision import datasets as tv_datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def safe_mkdir(target_dir):
    root, ext = os.path.splitext(target_dir)
    # is a file
    if len(ext) > 0:
        target_dir = os.path.split(root)
    else:
        target_dir = f"{target_dir}/"
    # if os.path.isfile(target_dir):
    #     target_dir,_ = os.path.split(target_dir.split('.')[0])
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)


def get_fashion_mnist(batch_size=64):
    training_loader = DataLoader(
        tv_datasets.FashionMNIST(
            root="data", train=True, download=True, transform=ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        tv_datasets.FashionMNIST(
            root="data", train=False, download=True, transform=ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return training_loader, test_loader


def uri_to_path(uri):
    parsed_uri = urlparse(uri)
    path = unquote(parsed_uri.path)
    return path


def is_cacheable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def get_model_size(model, as_MB=True):
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


def quantize(n):

    if float(n) < 0.1:
        ind = 2
        while str(n)[ind] == "0":
            ind += 1
        # print(ind)
        c = np.around(float(n), ind - 1)
    elif float(n) < 1.0:
        c = np.around(float(n), 2)
    elif float(n) < 10.0:
        c = int(n)
    else:
        c = int(2 ** np.round(math.log(float(n)) / math.log(2)))

    return c


def convert_to_scientific(n):
    return f"{float(n):0.0e}"


def closest_power_of_2(number):
    # Handle negative numbers by taking the absolute value
    number = abs(number)

    # Find the exponent (log base 2)
    exponent = math.log2(number)

    # Round the exponent to the nearest integer
    rounded_exponent = round(exponent)

    # Calculate the closest power of 2
    closest_value = 2 ** rounded_exponent

    return closest_value


def _is_number(n):
    # if isinstance(n,list):
    #     return all([_is_number(num) for num in n])
    try:
        float(n)  # Type-casting the string to `float`.
        # If string is not a valid `float`,
        # it'll raise `ValueError` exception
    # except ValueError, TypeError:
    except:
        return False
    return True


def quantize_dict(base_dict, quantize_fn=quantize):
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


def _discretize(n):

    if float(n) < 0.1:
        ind = 2
        while str(n)[ind] == "0":
            ind += 1
        # print(ind)
        c = np.around(float(n), ind - 1)
    elif float(n) < 1.0:
        c = np.around(float(n), 2)
    elif float(n) < 10.0:
        c = int(n)
    else:
        c = int(2 ** np.round(math.log(float(n)) / math.log(2)))
    return c


def discretize(n: list[float, int]) -> list[float, int]:
    """
    Discretize a list of inputs
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


def test_discretize():

    n = 0.234
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 0.00234
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 2.34
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 30143215
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = [3.3, 32144321, 0.032]
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = (1, 23)
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = "test"
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))

    n = 0.0005985885113477707
    n = str(n)
    print("input = {}, discretize(input)= {}".format(n, discretize(n)))


def apply_discretize_to_summary(text, info):

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

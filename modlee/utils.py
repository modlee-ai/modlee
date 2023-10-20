import json
from urllib.parse import urlparse, unquote
import os
import pathlib
import pickle
import requests

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
            root='data',
            train=True,
            download=True,
            transform=ToTensor(),
        ), batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        tv_datasets.FashionMNIST(
            root='data',
            train=False,
            download=True,
            transform=ToTensor(),
        ), batch_size=batch_size, shuffle=True,
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
    model_size = param_size+buffer_size
    if as_MB:
        model_size /= (1024**2)
    return model_size
        

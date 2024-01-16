"""
Test the recommender in the package, which acts as a client for the recommender system on the server.
Note that the recommender server must be running, either locally, and modify SERVER_ENDPOINT accordingly.
"""
SERVER_ENDPOINT = "http://127.0.0.1:6060"


import pytest
import modlee
modlee.init(api_key='modleemichael')
from modlee.recommender import Recommender
from torchvision import datasets as tv_datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_dataloader(batch_size=64):
    training_loader = DataLoader(
        tv_datasets.CIFAR10(
            root='data',
            train=True,
            download=True,
            transform=ToTensor(),
        ), batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        tv_datasets.CIFAR10(
            root='data',
            train=False,
            download=True,
            transform=ToTensor(),
        ), batch_size=batch_size, shuffle=True,
    )
    return training_loader, test_loader


def test_image_classification_recommender():
    dataloader, _ = get_dataloader()
    
    recommender = modlee.recommender.ImageClassificationRecommender(
        endpoint = SERVER_ENDPOINT
    )
    # alternatively,
    # recommender = Recommender.from_modality_task('image','classification')
    
    recommender.fit(dataloader)
    # make sure that the dataloader yields data that matches the recommender's expected format
    
    model = recommender.model
    
    model.forward(next(iter(dataloader))[0])
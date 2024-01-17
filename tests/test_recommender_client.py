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
dataloader,_ = get_dataloader()

@pytest.mark.parametrize('modality',['image'])
@pytest.mark.parametrize('task',['classification','segmentation'])
# @pytest.mark.parametrize('task',['segmentation'])
def test_image_recommenders(modality, task):

    # recommender = modlee.recommender.ImageClassificationRecommender(
    #     endpoint = SERVER_ENDPOINT
    # )
    # alternatively,
    recommender = modlee.recommender.from_modality_task(
        modality, task,
        endpoint=SERVER_ENDPOINT)
    
    recommender.fit(dataloader)
    
    # Get the model and pass an input through it
    model = recommender.model    
    x = next(iter(dataloader))[0]
    # breakpoint()
    # import torchvision
    
    # x = torchvision.transforms.Resize((300,300))(x)
    # breakpoint()
    y = model.forward(x)
    # breakpoint()
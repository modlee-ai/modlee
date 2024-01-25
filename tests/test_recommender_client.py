"""
Test the recommender in the package, which acts as a client for the recommender system on the server.
Note that the recommender server must be running, either locally, and modify SERVER_ENDPOINT accordingly.
"""
SERVER_ENDPOINT = "http://127.0.0.1:6060"


import pytest
import modlee
# modlee.init(api_key='modleemichael')
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
    
def test_recommended_model():
    # modlee.init(api_key='modleemichael')
    # load a pretrained model
    import torchvision
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.train()
    # breakpoint()
    
    # wrap into recommended model
    from modlee.recommender import RecommendedModel
    model = RecommendedModel(model)
    
    # load data
    train_loader, val_loader = get_dataloader()
    
    # breakpoint()
    # train
    model.train()
    import lightning
    with modlee.start_run() as run:
        trainer = lightning.pytorch.Trainer(max_epochs=3)
        trainer.fit(model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
            )
        breakpoint()
    
    # assert that the model has updated
    # assert that the loss has decreased
    breakpoint()
    pass
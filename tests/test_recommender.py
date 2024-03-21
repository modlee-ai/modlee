"""
Test the recommender in the package, which acts as a client for the recommender system on the server.
Note that the recommender server must be running, either locally, and modify RECOMMENDER_ORIGIN accordingly.
"""


import pytest
import modlee

# modlee.init(api_key='modleemichael')
from modlee.recommender import Recommender
from modlee.model import RecommendedModel
from modlee.config import RECOMMENDER_ORIGIN
import torch, torchvision, lightning
from torchvision import datasets as tv_datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_dataloader(batch_size=64):
    training_loader = DataLoader(
        tv_datasets.CIFAR10(
            root="data", train=True, download=True, transform=ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        tv_datasets.CIFAR10(
            root="data", train=False, download=True, transform=ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return training_loader, test_dataloader


train_dataloader, test_dataloader = get_dataloader()


class TestRecommender:
    @pytest.mark.parametrize("modality", ["image"])
    @pytest.mark.parametrize("task", ["classification", "segmentation"])
    def test_image_recommenders(self, modality, task):

        # recommender = modlee.recommender.ImageClassificationRecommender(
        #     endpoint = RECOMMENDER_ORIGIN
        # )
        # alternatively,
        recommender = modlee.recommender.from_modality_task(modality, task)

        recommender.fit(test_dataloader)

        # Get the model and pass an input through it
        model = recommender.model

        x, y_tgt = next(iter(test_dataloader))
        # breakpoint()
        # import torchvision

        # x = torchvision.transforms.Resize((300,300))(x)
        # breakpoint()
        y = model.forward(x)
        # loss = torch.nn.CrossEntropyLoss()(y, y_tgt)
        # TODO - test passing through loss, maybe run one training epoch
        # breakpoint()

    @pytest.mark.training
    def test_recommended_model(self,):
        # modlee.init(api_key='modleemichael')
        # load a pretrained model
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.train()
        # breakpoint()

        # wrap into recommended model
        model = RecommendedModel(model)

        model.train()
        with modlee.start_run() as run:
            trainer = lightning.pytorch.Trainer(max_epochs=1)
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader,
            )
            # breakpoint()

        # assert that the model has updated
        # assert that the loss has decreased
        # breakpoint()
        pass

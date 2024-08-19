"""
Test the recommender in the package, which acts as a client for the recommender system on the server.
Note that the recommender server must be running, either locally, and modify RECOMMENDER_ORIGIN accordingly.
"""

import pytest
import os, sys
TEST_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, TEST_ROOT)
from . import conftest
import modlee
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
    def test_calculate_metafeatures(
        self,
    ):
        recommender = modlee.recommender.Recommender()
        recommender.calculate_metafeatures(train_dataloader)

    def _test_modality_task(self, modality, task, kwargs):
        rec = modlee.recommender.from_modality_task(modality, task)
        assert (
            type(rec).__name__
            == f"{modality.capitalize()}{task.capitalize()}Recommender"
        )

    @pytest.mark.parametrize("args", conftest.IMAGE_MODALITY_TASK_KWARGS)
    def test_image_modality_task(self, args):
        self._test_modality_task(*args)
        
    def test_calculate_metafeatures_image(
        self,
    ):
        recommender = modlee.recommender.Recommender()
        base_metafeatures = recommender.calculate_metafeatures(train_dataloader)
        image_recommender = modlee.recommender.from_modality_task(
            modality="image", task="classification"
        )
        image_metafeatures = image_recommender.calculate_metafeatures(train_dataloader)

    @pytest.mark.training
    @pytest.mark.parametrize(
        "modality, task, kwargs", conftest.IMAGE_MODALITY_TASK_KWARGS
    )
    def test_image_recommenders(self, modality, task, kwargs):
        recommender = modlee.recommender.from_modality_task(modality, task, **kwargs)
        recommender.fit(test_dataloader)
        model = recommender.model
        x, y_tgt = next(iter(test_dataloader))
        y = model.forward(x)

    @pytest.mark.training
    def test_recommended_model(
        self,
    ):
        # load a pretrained model
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.train()

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
        pass

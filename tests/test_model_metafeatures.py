import pytest
from functools import partial
import copy

# from . import conftest
try:
    import conftest
except:
    from . import conftest

IMAGE_MODELS = conftest.IMAGE_MODELS

import torch
from torch import nn
import torchvision
import modlee
from modlee import model_metafeatures as mmf


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=10, output_size=10):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


MODEL = NeuralNetwork()


class TestModelMetafeatures:

    def _test_modality_task_model(self, modality, task, model):
        model_mf = mmf.from_modality_task(modality, task, torch_model=model)
        assert (
            type(model_mf).__name__
            == f"{modality.capitalize()}{task.capitalize()}ModelMetafeatures"
        )

        self._check_has_metafeatures(model_mf)

    for modality in conftest.MODALITIES:

        def _func(
            self,
            modality,
            task,
            model,
        ):
            return self._test_modality_task_model(
                modality,
                task,
                model,
            )

        _f = pytest.mark.parametrize(
            "modality, task, model",
            getattr(conftest, f"{modality.upper()}_MODALITY_TASK_MODEL"),
        )(_func)
        locals().update({f"test_{modality}_modality_task_model": _f})

    _check_has_metafeatures = partial(
        conftest._check_has_metafeatures, metafeature_types={"embedding", "properties"}
    )

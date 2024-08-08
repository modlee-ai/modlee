import pytest
from torch.utils.data import DataLoader
import mlflow
import lightning.pytorch as pl
import modlee
from modlee.model import callbacks, ModleeModel, SimpleModel

class TestModel:
    model = SimpleModel()
    dataloader = DataLoader(model.dataset)

    def test_check_step(self):
        # Training step should be defined
        assert self.model._check_step_defined("training_step")
        # Validation step shoudl not be defined
        assert not self.model._check_step_defined("validation_step")


    @pytest.mark.parametrize("modality", ["image"])
    @pytest.mark.parametrize("task_kwargs", [("classification", {"num_classes":10})])
    def test_modality_task(self, modality, task_kwargs):
        task, kwargs = task_kwargs
        model = modlee.model.from_modality_task(
            modality=modality,
            task=task, 
            **kwargs
        )
        assert model.modality == modality
        assert model.task == task
import pytest
import os, sys

TEST_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, TEST_ROOT)

from torch.utils.data import DataLoader
import mlflow
import lightning.pytorch as pl
import modlee
from modlee.model import callbacks, ModleeModel, SimpleModel
from . import conftest
from .conftest import model_from_args


class TestModel:
    model = SimpleModel()
    dataloader = DataLoader(model.dataset)

    def test_check_step(self):
        # Training step should be defined
        assert self.model._check_step_defined("training_step")
        # Validation step shoudl not be defined
        assert not self.model._check_step_defined("validation_step")

    # @pytest.mark.parametrize(
    #     "modality, task, kwargs", conftest.IMAGE_MODALITY_TASK_KWARGS
    # )
    # def test_image_modality_task(self, modality, task, kwargs):
    #     self._test_modality_task(modality, task, kwargs)
    def _test_modality_task(self, modality, task, kwargs):
        # model = model_from_args(modality_task_kwargs)
        model = modlee.model.from_modality_task(modality, task, **kwargs)
        assert model.modality == modality
        assert model.task == task

        dmf = model._get_data_metafeature_class()
        dmf_type = dmf
        model_dmf = f"{modality.capitalize()}DataMetafeatures"
        assert dmf_type.__name__ == model_dmf

        dmf_callback = model.data_metafeatures_callback
        assert dmf_callback.DataMetafeatures.__name__ == model_dmf

        model_mmf = f"{modality.capitalize()}{task.capitalize()}ModelMetafeatures"
        mmf_callback = model.model_metafeatures_callback
        assert mmf_callback.ModelMetafeatures.__name__ == model_mmf

    test_image_modality_task = pytest.mark.parametrize(
        "modality, task, kwargs", conftest.IMAGE_MODALITY_TASK_KWARGS
    )(_test_modality_task)

    def test_no_modality_task( self, ):
        self._test_modality_task("","", {})

    @pytest.mark.parametrize(
        "model", conftest.IMAGE_MODALITY_TASK_KWARGS, indirect=["model"]
    )
    def test_modality_task_indirect(self, model):
        # breakpoint()
        pass

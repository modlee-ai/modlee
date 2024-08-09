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


    # @pytest.mark.parametrize("modality", ["image"])
    @pytest.mark.parametrize("modality_task_kwargs", [
        ("image", "classification", {"num_classes":10}),
        ("image", "segmentation", {"num_classes":10}),
        ])
    def test_modality_task(self, modality_task_kwargs):
        modality, task, kwargs = modality_task_kwargs
        model = modlee.model.from_modality_task(
            modality=modality,
            task=task, 
            **kwargs
        )
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
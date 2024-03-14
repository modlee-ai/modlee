import pytest
import os

from torch.utils.data import DataLoader
import mlflow
import lightning.pytorch as pl
# import model
import modlee
from modlee.model import callbacks, ModleeModel, SimpleModel


class TestCallback:
    # Initialize modlee
    # Should not require setting API key
    model = SimpleModel()
    dataloader = DataLoader(model.dataset)
    # 
    @pytest.mark.training
    def test_transform_logging_callback(self):
        # create callback object
        cb = callbacks.LogTransformsCallback()
        # self.model.configure_callbacks = lambda : self.model.configure_callbacks()+[cb]
        # breakpoint()
        img_dataloader,_ = modlee.utils.get_fashion_mnist()
        
        with modlee.start_run() as run:
            trainer = modlee.Trainer(
                max_epochs=1,
                callbacks=self.model.configure_callbacks()+[cb])
            # Training will fail because the dataset is not matched to the model,
            # But just needs to get to on_train_start to activate the callback
            try:
                trainer.fit(
                    model=self.model,
                    train_dataloaders=img_dataloader
                )
            except:
                pass
        assert 'transforms.txt' in os.listdir(os.path.join(modlee.last_run_path(), 'artifacts'))

    @pytest.mark.training
    def test_checkpoint_callback(self):
        with modlee.start_run() as cur_run:
            trainer = modlee.Trainer(
                max_epochs=3)
            trainer.fit(
                model=self.model,
                train_dataloaders=self.dataloader
            )
        # Check that there is a checkpoint saved
        last_run_path = modlee.last_run_path()
        # artifacts_path = os.path.join()
        checkpoints_path = os.path.join(
            last_run_path, 'artifacts', 'lightning_logs','version_0','checkpoints')
        
        # Check that a checkpoint was saved
        assert len(os.listdir(checkpoints_path)) > 0
        

import pytest
import os, random
import mlflow
import lightning.pytorch as pl
import modlee
from modlee.model import callbacks
from modlee.model import ModleeCallback
from modlee.model import callbacks, ModleeModel, SimpleModel
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

SIZES = list(range(1,5))

class MIMODataset:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs =n_outputs
        
    def __len__(self):
        return 100
        
    def __getitem__(self, index):
        def generate_random(n): 
            return [torch.randn(*random.choices(SIZES, k=random.choice(SIZES))) for _ in range(n)]
        x = generate_random(self.n_inputs)
        y = generate_random(self.n_outputs)
        return [*x, *y]

class MIMOModel(ModleeModel):
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dataloader = DataLoader(MIMODataset(n_inputs, n_outputs))
        input_str = ','.join([f"x{i}" for i in range(n_inputs)])
        exec(f"self.forward = lambda {input_str}: self.model({input_str})")
    
    # def forward(self, *args):
        # return self.model(*args)
        

    def _step(self, batch, batch_idx):
        x, y = batch[:self.n_inputs], batch[self.n_inputs:]
        y_out = self(x)
        loss = F.mse_loss(y_out, y)
        return loss

    def training_step(self, *args, **kwargs):
        return {"loss": self._step(*args, **kwargs)}

    def validation_step(self, *args, **kwargs):
        return {"loss": self._step(*args, **kwargs)}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=0.001,
            momentum=0.9
        )

class TestCallback:
    # Initialize modlee
    # Should not require setting API key
    model = SimpleModel()
    dataloader = DataLoader(model.dataset)

    @pytest.mark.training
    def test_transform_logging_callback(self):
        # create callback object
        cb = callbacks.LogTransformsCallback()
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
            last_run_path, 'artifacts', 'checkpoints','train')
        # Check that a checkpoint was saved
        assert len(os.listdir(checkpoints_path)) > 0

    def test_data_metafeatures_callback(self, ):
        # TODO - modularize for any type of ModleeModel
        dmf_callback = callbacks.DataMetafeaturesCallback()
        dmf_callback._log_data_metafeatures_dataloader(self.dataloader)
        
    @pytest.mark.parametrize("n_inputs", list(range(1,4)))
    @pytest.mark.parametrize("n_outputs", list(range(1,4)))
    def test_get_input(self, n_inputs, n_outputs):
        model = MIMOModel(n_inputs, n_outputs)
        batch = next(iter(model.dataloader))
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=1)
            callback_input = ModleeCallback().get_input(
                trainer, model, model.dataloader
            )
            assert len(callback_input)==n_inputs

    # def test_log_modality_task()
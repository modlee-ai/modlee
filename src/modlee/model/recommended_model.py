from modlee.model import ModleeModel
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler


class RecommendedModel(ModleeModel):
    """
    A ready-to-train ModleeModel that wraps around a recommended model from the recommender.
    Contains default functions for training.
    """

    def __init__(self, model, loss_fn=F.cross_entropy, *args, **kwargs):
        """
        Constructor for a recommended model.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        y_out = self(x)
        loss = self.loss_fn(y_out, y)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx, *args, **kwargs):
        x, y = val_batch
        y_out = self(x)
        loss = self.loss_fn(y_out, y)
        return {"val_loss": loss}

    def configure_optimizers(
        self,
    ):
        """
        Configure a default AdamW optimizer with learning rate decay.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.8, patience=5
        )
        return optimizer

    def on_train_epoch_end(self) -> None:
        """
        Update the learning rate scheduler.
        """
        sch = self.scheduler
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["loss"])
            self.log("scheduler_last_lr", sch._last_lr[0])
        return super().on_train_epoch_end()

    def configure_callbacks(self):
        base_callbacks = super().configure_callbacks()
        return base_callbacks

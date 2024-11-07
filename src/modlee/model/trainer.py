import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from urllib.parse import urlparse
import copy

import modlee
from modlee.model.callbacks import LRSchedulerCallback,LossLoggingCallback,CustomModelCheckpoint

class AutoTrainer:

    def __init__(self, max_epochs=1):
        """
        Initialize the AutoTrainer with a model.
        
        :param model: The Lightning model to train.
        """
        self.max_epochs = max_epochs
        self.model = None
        self.best_model_weights = None
        self.best_epoch = None
        self.best_loss = float('inf')

    def fit(self, model=None, train_dataloaders=None, val_dataloaders=None):
        """
        Train the model with early stopping, learning rate scheduling, and
        weight reinitialization based on validation or training loss.

        :param train_dataloader: The training dataloader (required).
        :param max_epochs: The maximum number of epochs to train for.
        :param val_dataloader: The validation dataloader (optional).
        """

        self.model = model

        print("----------------------------------------------------------------")
        print("Modlee AutoTrainer:")
        print(" - training your model ...")
        # print("     - Running this model: {}".format("./model.py"))
        print("----------------------------------------------------------------")

        # Define the callbacks list
        callbacks = self.model.configure_callbacks()

        # If val_dataloader is provided, monitor val_loss, otherwise monitor train_loss
        if val_dataloaders is not None:
            # Early stopping callback (monitor validation loss)
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=10, verbose=True, mode="min"
            )
            # ModelCheckpoint to save the best model based on validation loss
            custom_checkpoint_callback = CustomModelCheckpoint(monitor="val_loss", mode="min", verbose=True)
        else:
            # Early stopping callback (monitor training loss if no validation set)
            early_stopping = EarlyStopping(
                monitor="loss", patience=10, verbose=True, mode="min"
            )
            # ModelCheckpoint to save the best model based on training loss
            custom_checkpoint_callback = CustomModelCheckpoint(monitor="loss", mode="min", verbose=True)
        
        callbacks.append(early_stopping)        
        callbacks.append(custom_checkpoint_callback)

        # Custom learning rate scheduler callback (without modifying the optimizer)
        lr_scheduler_callback = LRSchedulerCallback(patience=5, factor=0.95, min_lr=1e-6)
        callbacks.append(lr_scheduler_callback)

        # Additional callback to monitor learning rate
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

        # Add the custom loss logging callback
        # loss_logging_callback = LossLoggingCallback()
        # callbacks.append(loss_logging_callback)


        # Training loop with model reinitialization
        with modlee.start_run() as run:
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                enable_model_summary=False,
                log_every_n_steps=100,  # Customize logging frequency
            )

            # Training the model
            trainer.fit(
                model=self.model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,  # This can be None
            )

""" 
Recommender for models.
"""
import json
import requests
import logging
import modlee
from modlee.utils import get_model_size, typewriter_print
from modlee.converter import Converter
from datetime import datetime
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from time import sleep
import sys
import os
from urllib.parse import urlparse
from modlee import ModleeClient
import modlee

API_KEY = os.environ.get("MODLEE_API_KEY", 'None')
modlee_converter = Converter()
modlee_client = ModleeClient(api_key=API_KEY)
SERVER_ENDPOINT = modlee_client.origin

class Recommender(object):
    """
    Recommender for models conditioned on datasets.
    """
    def __init__(
        self, dataloader=None, origin=SERVER_ENDPOINT, *args, **kwargs
    ) -> None:
        """ 
        Constructor for recommender.
        
        :param dataloader: The dataloader to analyze, defaults to None.
        :param origin: The origin (scheme://hostname:port) for the server, defaults to Modlee's server. 
        """
        self._model = None
        self.modality = None
        self.task = None
        self.metafeatures = None
        self.origin = origin
        if dataloader is not None:
            self.analyze(dataloader)

    def __call__(self, *args, **kwargs):
        """
        Wrapper to analyze
        """
        self.analyze(*args, **kwargs)

    def analyze(self, dataloader=None, *args, **kwargs):
        """
        Analyze a dataloader and calculate data metafeatures.

        :param dataloader: The dataloader to analyze. If not given, tries to use the class dataloader.
        """
        if not dataloader:
            dataloader = self.dataloader
        self.dataloader = dataloader
        if not dataloader:
            raise Exception(f'Dataloader not provided and not previously set.')
        self.metafeatures = self.calculate_metafeatures(dataloader)
        logging.info("Finished analyzing dataset.")

    fit = analyze # Alias for the analyze method

    def calculate_metafeatures(self, dataloader, data_metafeature_cls=modlee.data_metafeatures.DataMetafeatures):
        """
        Calculate metafeatures.

        :param dataloader: The dataloader on which to calculate metafeatures.
        :return: The metafeatures of the data as a dictionary.
        """
        if modlee.data_metafeatures.module_available:
            logging.info("Analyzing dataset based on data metafeatures...")

            return data_metafeature_cls(dataloader, testing=True).stats_rep
        else:
            print("Could not analyze data (check access to server)")
            return {}

    def _get_model_text(self, metafeatures):
        """
        Get the text for a recommended model based on data metafeatures.
        Sends the metafeatures to the server, the server analyzes the metafeatures
        and returns a client-parseable text representation of the model.

        :param metafeatures: The data metafeatures to send to the server.
        :return: The model as text that can be parsed into a trainable object. 
        """
        assert (
            self.modality is not None
        ), "Recommender modality is not set (e.g. image, text)"
        assert (
            self.task is not None
        ), "Recommender task is not set (e.g. classification, segmentation)"
        metafeatures = json.loads(json.dumps(metafeatures))
       
        res = modlee_client.get(
            path=f"model/{self.modality}/{self.task}",
            data=json.dumps({"data_features": metafeatures}),
            headers={"Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Methods": "*",
                    },
            timeout=20,
        )
        model_text = res.content
        return model_text

    @property
    def model(self):
        """ 
        The cached model.
        """
        if self._model is None:
            logging.info(
                "No model recommendation, call .analyze on a dataloader first."
            )
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
    
    def write_file(self, file_contents, file_path):
        """ 
        Helper function to write a file.
        
        :param file_contents: The contents to write.
        :param file_path: The path to the file.
        """
        with open(file_path, "w") as _file:
            _file.write(file_contents)

    def train(self, max_epochs=1, val_dataloaders=None):
        """ 
        Train the recommended model.
        
        :param max_epochs: The maximum epochs to train for.
        :param val_dataloaders: The validation dataloaders, optional.
        """

        print("----------------------------------------------------------------")
        print("Training your recommended modlee model:")
        print("     - Running this model: {}".format("./model.py"))
        print("     - On the dataloader previously analyzed by the recommender")
        print("----------------------------------------------------------------")

        callbacks = self.model.configure_callbacks()
        if val_dataloaders is not None:
            callbacks.append(
                pl.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, verbose=True
                )
            )
        with modlee.start_run() as run:
            trainer = pl.Trainer(
                max_epochs=max_epochs, callbacks=callbacks, enable_model_summary=False
            )
            trainer.fit(
                model=self.model,
                train_dataloaders=self.dataloader,
                val_dataloaders=val_dataloaders,
            )
            self.run_artifact_uri = urlparse(run.info.artifact_uri).path
            self.run_id = run.info.artifact_uri.split("/")[-2]
            self.exp_id = run.info.artifact_uri.split("/")[-3]
            self.run_folder = self.run_artifact_uri.split("///")[-1].split("artifacts")[
                0
            ]
   
    def get_input_torch(self):
        """ 
        Get an input from the dataloader.

        :return: A tuple of the inputs (tensors) and their sizes.
        """

        # Assuming you have a DataLoader called dataloader
        for batch in self.dataloader:
            # Access the first element in the batch
            one_element = batch
            break  # Exit the loop after processing the first batch

        input_sizes = [
            [1] + list(b.size()[1:])
            for i, b in enumerate(one_element)
            if i in self.dataloader_input_inds
        ]
        input_torches = [torch.rand(ins) for ins in input_sizes]
        return input_torches, input_sizes

    def get_code_text(self):
        """ 
        Get the code for a model as text (deprecated?).
        
        :return: The model code as text.
        """
        _get_code_text_for_model = getattr(modlee, "get_code_text_for_model", None)

        if _get_code_text_for_model is not None:
            # ==== METHOD 1 ====
            # Save model as code using parsing
            self.model_code = modlee.get_code_text_for_model(
                self.model, include_header=True
            )
        else:
            self.model_code = modlee_converter.onnx_text2code(self.model_onnx_text)

        try:
            self.model_code = self.model_code.replace("= model", "= " + self.model_str)
        except:
            pass
        self.model_code = self.model_code.replace("self, model,", "self,")
        return self.model_code
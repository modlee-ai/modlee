""" 
Recommender for models.
"""
import json
import requests
import logging

logging.basicConfig(level=logging.INFO)

import modlee
from modlee.utils import get_model_size, typewriter_print
from modlee.converter import Converter

modlee_converter = Converter()

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
API_KEY = os.environ.get("MODLEE_API_KEY", 'None')
import modlee

modlee_client = ModleeClient(api_key=API_KEY)
# SERVER_ENDPOINT = modlee_client.endpoint
SERVER_ENDPOINT = modlee_client.origin
#SERVER_ORIGIN = 'http://127.0.0.1:7070'
#SERVER_ENDPOINT = 'http://ec2-3-84-155-233.compute-1.amazonaws.com:7070'
#print(SERVER_ENDPOINT)

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
        # self.write_files()

    fit = analyze

    def calculate_metafeatures(self, dataloader):
        """
        Calculate metafeatures.

        :param dataloader: The dataloader on which to calculate metafeatures.
        :return: The metafeatures of the data as a dictionary.
        """
        if modlee.data_metafeatures.module_available:
            analyze_message = "[Modlee] -> Just a moment, analyzing your dataset ...\n"

            typewriter_print(analyze_message, sleep_time=0.01)

            # ??? Add in type writer print
            # TODO - generalize to a base DataMetafeatures,
            # override this method for modality-specific calclations
            return modlee.data_metafeatures.ImageDataMetafeatures(dataloader, testing=True).stats_rep
            # ??? Convert to ImageDataMetafeatures
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
        # breakpoint()
        # res = requests.get(
        #     f"{self.origin}/model/{self.modality}/{self.task}",
        #     data=json.dumps({"data_features": metafeatures}),
        #     headers={"Content-Type": "application/json"},
        #     verify=False,
        # )
        
        res = modlee_client.get(
            path=f"model/{self.modality}/{self.task}",
            data=json.dumps({"data_features": metafeatures}),
            headers={"Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Methods": "*",
                    # "X-API-KEY": API_KEY
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

    def get_model_details(self):
        """ 
        Get the details of a model with verbose logging.
        """
        # ??? save self.model_onnx_text and self.model_code to local place, point use to them here
        # In case you wanted to take a deeper look I saved the onnx graph summary here:, I also saved and python editable version of the model with train, val, and optimzers. This is a great place to start your own model exploration!

        print("--- Modlee Recommended Model Details --->")

        indent = "        "
        text_indent = "\n            "

        # summary_message = '\n[Modlee] -> In case you want to take a deeper look, I saved the summary of my current model recommendation here:{}file: {}'.format(text_indent+indent,self.model_onnx_text_file)
        summary_message = "\n[Modlee] -> In case you want to take a deeper look, I saved the summary of my current model recommendation here:{}file: {}".format(
            text_indent + indent, "./model.txt"
        )
        typewriter_print(summary_message, sleep_time=0.01)

        # code_message = '\n[Modlee] -> I also saved the model as a python editable version (model def, train, val, optimizer):{}file: {}{}This is a great place to start your own model exploration!'.format(text_indent+indent,self.model_code_file,text_indent)
        code_message = "\n[Modlee] -> I also saved the model as a python editable version (model def, train, val, optimizer):{}file: {}{}This is a great place to start your own model exploration!".format(
            text_indent + indent, "./model.py", text_indent
        )
        typewriter_print(code_message, sleep_time=0.01)

    def _write_files(self):
        """ 
        Write the model text and code to files.
        """
        self.model_onnx_text_file = "./model_summary.txt"
        self.model_code_file = "./model_code.py"
    
        if hasattr(self, "model_text"):
            with open(self.model_onnx_text_file, "w") as file:
                file.write(self.model_text)
        if hasattr(self, "model_code"):
            with open(self.model_code_file, "w") as file:
                file.write(self.model_code)

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
            # if val_dataloaders == None:
            #     trainer.fit(
            #         model=self.model,
            #         train_dataloaders=self.dataloader)
            # else:
            #     trainer.fit(
            #         model=self.model,
            #         train_dataloaders=self.dataloader,
            #         val_dataloaders=val_dataloaders)

            self.run_artifact_uri = urlparse(run.info.artifact_uri).path
            self.run_id = run.info.artifact_uri.split("/")[-2]
            self.exp_id = run.info.artifact_uri.split("/")[-3]
            self.run_folder = self.run_artifact_uri.split("///")[-1].split("artifacts")[
                0
            ]
            # <RunInfo: artifact_uri='file:///Users/brad/Github_Modlee/modlee_survey/notebooks/mlruns/0/e2d08510ac28438681203a930bb713ed/artifacts', end_time=None, experiment_id='0', lifecycle_stage='active', run_id='e2d08510ac28438681203a930bb713ed', run_name='skittish-trout-521', run_uuid='e2d08510ac28438681203a930bb713ed', start_time=1697907858055, status='RUNNING', user_id='brad'>

    def train_documentation_locations(self):
        """ 
        Print the location of documented assets.
        """

        vertical_sep = "\n-----------------------------------------------------------------------------------------------\n"
        path_indent = "        Path: "
        indent = "        "
        doc_indent = "                     "

        print(vertical_sep)

        print(
            "Modlee documented all the details about your trained model and experiment here: \n\n{}{}".format(
                path_indent, self.run_folder
            )
        )
        print(
            "{}Experiment_id: automatically assigned to | {}".format(
                indent, self.exp_id
            )
        )
        print("{}Run_id: automatically assigned to | {}".format(indent, self.run_id))

        print(vertical_sep)

    def train_documentation_shared(self):
        """ 
        Print the shared experiment assets. 
        """

        vertical_sep = "\n-----------------------------------------------------------------------------------------------\n"
        path_indent = "        Path: "
        indent = "        "
        doc_indent = "                     "

        print(vertical_sep)

        print(
            "Modlee auto-documents your experiment locally and learns from non-sensitive details:\n -> Sharing helps to enhance ML model recommendations across the entire community of modlee users, including you!\n"
        )

        print("Modlee's ML Experiment Documentation Overview: \n")
        print("[ Local ] [ Shared ] Documented Element Description ...")
        print(vertical_sep[2:-2])

        print("[       ] [        ] Dataloader\n")

        print(
            "[   X   ] [        ] Sampling of Dataloader: for your benefit, and in case we have improvements to our data analysis process"
        )
        print(
            "{}{}{}".format(
                doc_indent, path_indent, self.run_artifact_uri + "/model/snapshot*"
            )
        )

        print("[   X   ] [        ] Model Weights")
        print(
            "{}{}{}".format(
                doc_indent, path_indent, self.run_artifact_uri + "/model/data/model.pth"
            )
        )

        print(
            "[   X   ] [   X    ] Dataloader Complexity Analysis: Applying standard statistics (dims, mean, std, var, etc ...) & ML methods (clustering, etc ...) to your dataset"
        )
        print(
            "{}{}{}".format(
                doc_indent, path_indent, self.run_artifact_uri + "/stats_rep"
            )
        )

        print(
            "[   X   ] [   X    ] Modlee Model Code (model def, training step, validation step, optimizers)"
        )
        print(
            "{}{}{}".format(
                doc_indent, path_indent, self.run_artifact_uri + "/model.py"
            )
        )

        print("[   X   ] [   X    ] Experiment Metrics: (loss, accuracy, etc ...)")
        print("{}{}{}".format(doc_indent, path_indent, self.run_folder + "/metrics/"))

        print(vertical_sep)

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


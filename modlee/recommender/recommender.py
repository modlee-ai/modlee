
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

SERVER_ENDPOINT = 'http://ec2-3-84-155-233.compute-1.amazonaws.com:7070'
# SERVER_ENDPOINT = 'http://127.0.0.1:6060'


class Recommender(object):
    """
    Recommends models given a dataset

    Args:
        object (_type_): _description_
    """

    def __init__(self, dataloader=None, endpoint=SERVER_ENDPOINT, *args, **kwargs) -> None:
        self._model = None
        self.modality = None
        self.task = None
        self.meta_features = None
        self.endpoint = endpoint
        if dataloader is not None:
            self.analyze(dataloader)
        
    def __call__(self, *args, **kwargs):
        """
        Wrapper to analyze
        """
        self.analyze(*args, **kwargs)

        
    def analyze(self, dataloader, *args, **kwargs):
        """
        Set dataloader and calculate meta features

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader
        """
        self.dataloader = dataloader
        self.meta_features = self.calculate_meta_features(dataloader)
        # self.write_files()
    fit = analyze
    
    def calculate_meta_features(self, dataloader):
        if modlee.data_stats.module_available:
            analyze_message = "[Modlee] -> Just a moment, analyzing your dataset ...\n"

            typewriter_print(analyze_message,sleep_time=0.01)        

            #??? Add in type writer print
            return modlee.data_stats.ImageDataStats(dataloader, testing=True).stats_rep
            #??? Convert to ImageDataStats
        else:
            print("Could not analyze data (check access to server)")
            return {}

    def _get_model_text(self, meta_features):
        """Get the model text from the server, based on the data meta features

        :param meta_features: _description_
        :return: _description_
        """
        assert self.modality is not None, 'Recommender modality is not set (e.g. image, text)'
        assert self.task is not None, 'Recommender task is not set (e.g. classification, segmentation)'
        meta_features = json.loads(json.dumps(meta_features))
        res = requests.get(
            f'{self.endpoint}/model/{self.modality}/{self.task}',
            data=json.dumps({'data_features':meta_features}),
            headers={'Content-Type':'application/json'},
        )
        model_text = res.content
        return model_text
    
    @property
    def model(self):
        if self._model is None:
            logging.info(
                'No model recommendation, call .analyze on a dataloader first.')
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        
        
    def get_model_details(self):
        #??? save self.model_onnx_text and self.model_code to local place, point use to them here
        # In case you wanted to take a deeper look I saved the onnx graph summary here:, I also saved and python editable version of the model with train, val, and optimzers. This is a great place to start your own model exploration!
        
        print('--- Modlee Recommended Model Details --->')

        indent = '        '
        text_indent = '\n            '

        summary_message = '\n[Modlee] -> In case you want to take a deeper look, I saved the summary of my current model recommendation here:{}file: {}'.format(text_indent+indent,self.model_onnx_text_file)
        typewriter_print(summary_message,sleep_time=0.01)        

        code_message = '\n[Modlee] -> I also saved the model as a python editable version (model def, train, val, optimizer):{}file: {}{}This is a great place to start your own model exploration!'.format(text_indent+indent,self.model_code_file,text_indent)
        typewriter_print(code_message,sleep_time=0.01)        

    def _write_files(self):

        self.model_onnx_text_file = './modlee_model_summary.txt'
        self.model_code_file = './modlee_model_code.py'

        with open(self.model_onnx_text_file, 'w') as file:
            file.write(self.model_text)
        if hasattr(self,'model_code'):
            with open(self.model_code_file, 'w') as file:
                file.write(self.model_code)
                
    def write_file(self, file_contents, file_path):
        with open(file_path, 'w') as _file:
            _file.write(file_contents)


    def train(self,max_epochs=1,val_dataloaders=None):

        print('----------------------------------------------------------------')
        print('Training your recommended modlee model:')
        print('     - Running this model: {}'.format(self.model_code_file))
        print('     - On the dataloader previously analyzed by ImageClassificationRecommender')
        print('----------------------------------------------------------------')

        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=max_epochs)
            if val_dataloaders == None:
                trainer.fit(
                    model=self.model,
                    train_dataloaders=self.dataloader)
            else:
                trainer.fit(
                    model=self.model,
                    train_dataloaders=self.dataloader,
                    val_dataloaders=val_dataloaders)
                
            self.run_artifact_uri = urlparse(run.info.artifact_uri).path
            self.run_id = run.info.artifact_uri.split('/')[-2]
            self.exp_id = run.info.artifact_uri.split('/')[-3]
            self.run_folder = self.run_artifact_uri.split('///')[-1].split('artifacts')[0]
            # <RunInfo: artifact_uri='file:///Users/brad/Github_Modlee/modlee_survey/notebooks/mlruns/0/e2d08510ac28438681203a930bb713ed/artifacts', end_time=None, experiment_id='0', lifecycle_stage='active', run_id='e2d08510ac28438681203a930bb713ed', run_name='skittish-trout-521', run_uuid='e2d08510ac28438681203a930bb713ed', start_time=1697907858055, status='RUNNING', user_id='brad'>

    def train_documentation_locations(self):

        vertical_sep = "\n-----------------------------------------------------------------------------------------------\n"
        path_indent = "        Path: "
        indent = "        "
        doc_indent = "                     "

        print(vertical_sep)

        print('Modlee documented all the details about your trained model and experiment here: \n\n{}{}'.format(path_indent,self.run_folder))
        print('{}Experiment_id: automatically assigned to | {}'.format(indent,self.exp_id))
        print('{}Run_id: automatically assigned to | {}'.format(indent,self.run_id))

        print(vertical_sep)

    def train_documentation_shared(self):

        vertical_sep = "\n-----------------------------------------------------------------------------------------------\n"
        path_indent = "        Path: "
        indent = "        "
        doc_indent = "                     "

        print(vertical_sep)

        print('Modlee auto-documents your experiment locally and learns from non-sensitive details:\n -> Sharing helps to enhance ML model recommendations across the entire community of modlee users, including you!\n')

        print("Modlee's ML Experiment Documentation Overview: \n")
        print("[ Local ] [ Shared ] Documented Element Description ...")
        print(vertical_sep[2:-2])

        print("[       ] [        ] Dataloader\n")

        print("[   X   ] [        ] Sampling of Dataloader: for your benefit, and in case we have improvements to our data analysis process")
        print("{}{}{}".format(doc_indent,path_indent,self.run_artifact_uri+'/model/snapshot*'))


        print("[   X   ] [        ] Model Weights")
        print("{}{}{}".format(doc_indent,path_indent,self.run_artifact_uri+'/model/data/model.pth'))


        print("[   X   ] [   X    ] Dataloader Complexity Analysis: Applying standard statistics (dims, mean, std, var, etc ...) & ML methods (clustering, etc ...) to your dataset")
        print("{}{}{}".format(doc_indent,path_indent,self.run_artifact_uri+'/stats_rep'))


        print("[   X   ] [   X    ] Modlee Model Code (model def, training step, validation step, optimizers)")
        print("{}{}{}".format(doc_indent,path_indent,self.run_artifact_uri+'/model.py'))

        print("[   X   ] [   X    ] Experiment Metrics: (loss, accuracy, etc ...)")
        print("{}{}{}".format(doc_indent,path_indent,self.run_folder+'/metrics/'))

        print(vertical_sep)

    def get_input_torch(self):

        # Assuming you have a DataLoader called dataloader
        for batch in self.dataloader:
            # Access the first element in the batch
            one_element = batch
            break  # Exit the loop after processing the first batch

        input_sizes = [[1]+list(b.size()[1:]) for i,b in enumerate(one_element) if i in self.dataloader_input_inds]
        input_torches = [torch.rand(ins) for ins in input_sizes]

        return input_torches,input_sizes

    def get_code_text(self):
        _get_code_text_for_model = getattr(modlee, 'get_code_text_for_model', None)

        if _get_code_text_for_model is not None:
            # ==== METHOD 1 ====
            # Save model as code using parsing
            self.model_code = modlee.get_code_text_for_model(self.model, include_header=True)
        else:
            self.model_code = modlee_converter.onnx_text2code(self.model_onnx_text)

        try:
            self.model_code = self.model_code.replace('= model','= '+self.model_str)
        except:
            pass
        self.model_code = self.model_code.replace('self, model,','self,')

        return self.model_code

class DumbRecommender(Recommender):
    def __init__(self) -> None:
        super().__init__()


class ActualRecommender(Recommender):
    def __init__(self) -> None:
        super().__init__()

class ModelSummaryRecommender(Recommender):
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)
        
    def analyze(self, dataloader, *args, **kwargs):
        super().analyze(dataloader, *args, **kwargs)
        num_classes = len(dataloader.dataset.classes)
        self.meta_features.update({
            'num_classes': num_classes
        })
        # try:
        if 1:
            self.model_onnx_text = self._get_onnx_text(self.meta_features)
            model = modlee_converter.onnx_text2torch(self.model_onnx_text)
            for param in model.parameters():
                # torch.nn.init.constant_(param,0.001)
                try:
                    torch.nn.init.xavier_normal_(param,1.0)
                except:
                    torch.nn.init.normal_(param)
            model = self._append_classifier_to_model(model, num_classes)
            self.model = ImageClassificationModleeModel(model)

            self.get_code_text()
            self.model_onnx_text = self.model_onnx_text.decode('utf-8')
            clean_model_onnx_text = '>'.join(self.model_onnx_text.split('>')[1:])
            typewriter_print(clean_model_onnx_text,sleep_time=0.005)
            self.write_files()
            
        # except:
        else:
            print("Could not retrieve model, data features may be malformed ")
            self.model = None
        
    def _get_onnx_text(self, meta_features):
        meta_features = json.loads(json.dumps(meta_features))
        res = requests.post(
            f'{SERVER_ENDPOINT}/infer',
            data={'data_stats':str(meta_features)}
        )
        onnx_text = res.content
        return onnx_text
    
    def _append_classifier_to_model(self,model,num_classes):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.model_clf_layer = nn.Linear(1000, num_classes)

            def forward(self, x):
                x = self.model(x)
                x = self.model_clf_layer(x)
                return x
        return Model()

        
class RecommendedModel(modlee.modlee_model.ModleeModel):
# class RecommendedModel(pl.LightningModule):
    """
    A ready-to-train ModleeModel that wraps around a recommended model
    Defines a basic training pipeline

    Args:
        modlee (_type_): _description_
    """
    def __init__(self, model, loss_fn=F.cross_entropy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        y_out = self(x)
        loss = self.loss_fn(y_out, y)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx, *args, **kwargs):
        x, y = val_batch
        y_out = self(x)
        loss = self.loss_fn(y_out, y)
        return {'val_loss':loss}
    
    def configure_optimizers(self,):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.001,
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.8,
            patience=10,
        )
        return optimizer
    
    def on_train_epoch_end(self) -> None:
        """
        Update the learning rate scheduler
        """
        sch = self.scheduler
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["loss"])
            self.log('scheduler_last_lr',sch._last_lr[0])
        return super().on_train_epoch_end()
    
    def configure_callbacks(self):
        base_callbacks = super().configure_callbacks()
        base_callbacks.append(
            pl.callbacks.EarlyStopping(
                'val_loss',
                patience=10,
                verbose=True,)
        )
        return base_callbacks
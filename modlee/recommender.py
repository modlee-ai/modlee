
from torchvision.models import \
    resnet34, ResNet34_Weights, \
    resnet18, ResNet18_Weights, \
    resnet152, ResNet152_Weights
import torchvision
from torch.nn import functional as F
from torch import nn
import torch
import logging

import modlee
from modlee.converter import Converter
from modlee.utils import get_model_size
modlee_converter = Converter()
logging.basicConfig(level=logging.INFO)

from datetime import datetime

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np

from time import sleep
import sys

import os
from urllib.parse import urlparse


class Recommender(object):
    """
    Recommends models given a dataset

    Args:
        object (_type_): _description_
    """

    def __init__(self) -> None:
        self._model = None
        self.meta_features = None

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

    @property
    def model(self):
        if self._model is None:
            logging.info(
                'No model recommendation, call .analyze on a dataloader first.')
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


class DumbRecommender(Recommender):
    def __init__(self) -> None:
        super().__init__()


class ActualRecommender(Recommender):
    def __init__(self) -> None:
        super().__init__()


class ImageRecommender(Recommender):
    def __init__(self):
        super().__init__()



def typewriter_print(text,sleep_time=0.001,max_line_length=150,max_lines=50):

   text_lines = text.split('\n')

   if len(text_lines)>max_lines:
      text_lines = text_lines[:max_lines]+['...\n']

   def shorten_if_needed(line,max_line_length):
      if len(line)>max_line_length:
         return line[:max_line_length]+' ...\n'
      else:
         return line+'\n'

   text_lines = [shorten_if_needed(l,max_line_length) for l in text_lines]

   for line in text_lines:
      for c in line:
         print(c, end='')
         sys.stdout.flush()
         sleep(sleep_time)
        
# typewriter_print(text,sleep_time=0.005)


class ImageClassificationRecommender(ImageRecommender):
    def __init__(self,dataloader,max_model_size_MB=10,num_classes=10,dataloader_input_inds=[0],min_accuracy=None,max_loss=None):
        super().__init__()

        sleep(0.5)

        print('---Contacting Modlee for a Recommended Image Classification Model--->\n')
        
        sleep(0.5)

        self.meta_features = self.calculate_meta_features(dataloader)

        #??? type writer effect
        generation_message = '[Modlee] -> From analyzing your dataset, I recommend the following neural network model:\n'
        typewriter_print(generation_message,sleep_time=0.01)        


        self.dataloader = dataloader
        self.max_model_size_MB = max_model_size_MB
        self.num_classes = num_classes
        self.dataloader_input_inds = dataloader_input_inds

        self.input_torches,self.input_sizes = self.get_input_torch()

        #sloppy for now
        self.input_torches = self.input_torches[0]
        # print(self.input_torches)
        self.input_sizes = self.input_sizes[0]
        # print(self.input_sizes)

        self.num_classes = num_classes


        self.meta_features.update({
            'num_classes': num_classes,
            'input_sizes':self.input_sizes,
        })

        self.model_torch,self.model_str = self.recommend_model(self.meta_features)
        self.model_onnx = modlee_converter.torch2onnx(self.model_torch, input_dummy=self.input_torches)
        self.model_onnx_text = modlee_converter.onnx2onnx_text(self.model_onnx)
        self.model = ImageClassificationModleeModel(self.model_torch)

        _get_code_text_for_model = getattr(modlee, 'get_code_text_for_model', None)

        if _get_code_text_for_model is not None:
            # ==== METHOD 1 ====
            # Save model as code using parsing
            self.model_code = modlee.get_code_text_for_model(self.model, include_header=True)
        else:
            self.model_code = modlee_converter.onnx_text2code(self.model_onnx_text)

        self.model_code = self.model_code.replace('= model','= '+self.model_str)
        self.model_code = self.model_code.replace('self, model,','self,')

        clean_model_onnx_text = '>'.join(self.model_onnx_text.split('>')[1:])

        typewriter_print(clean_model_onnx_text,sleep_time=0.005)        

        self.model_onnx_text_file = './modlee_model_summary.txt'
        self.model_code_file = './modlee_model_code.py'

        with open(self.model_onnx_text_file, 'w') as file: file.write(self.model_onnx_text)
        with open(self.model_code_file, 'w') as file: file.write(self.model_code)



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


        print('Modlee auto-documents your experiment locally and learns from non-sensitive details: sharing helps to enhance ML model recommendations across the entire community of modlee users.')

        vertical_sep = "\n-----------------------------------------------------------------------------------------------\n"
        path_indent = "        Path: "
        indent = "        "
        doc_indent = "                     "

        print(vertical_sep)


        print("Modlee's ML Experiment Documentation Overview: Modlee automatically saves details from modlee_model.train() locally and shares some elements with itself to improve recommendations for the community, including you!\n")
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


    def get_input_torch(self):

        # Assuming you have a DataLoader called dataloader
        for batch in self.dataloader:
            # Access the first element in the batch
            one_element = batch
            break  # Exit the loop after processing the first batch

        input_sizes = [[1]+list(b.size()[1:]) for i,b in enumerate(one_element) if i in self.dataloader_input_inds]
        input_torches = [torch.rand(ins) for ins in input_sizes]

        return input_torches,input_sizes



    def recommend_model(self, meta_features):
        """
        Recommend a model based on meta-features

        Args:
            meta_features (_type_): A dictionary of meta-features

        Returns:
            torch.nn.Module: The recommended model
        """

        num_layers=1
        num_channels=8

        ret_model = VariableConvNet(num_layers,num_channels,self.input_sizes,self.num_classes)
        model_str = 'VariableConvNet({},{},{},{})'.format(num_layers,num_channels,self.input_sizes,self.num_classes)

        for i in range(10):

            model = VariableConvNet(int(num_layers),int(num_channels),self.input_sizes,self.num_classes)

            if get_model_size(model)<self.max_model_size_MB:
                ret_model = model
                num_layers += 1
                num_channels = num_channels*2
            else:
                break

        return ret_model,model_str


class VariableConvNet(nn.Module):
    def __init__(self, num_layers, num_channels, input_sizes_orig, num_classes):
        super(VariableConvNet, self).__init__()

        layers = []  # List to hold convolutional layers
        
        input_sizes = input_sizes_orig[1:]#this is a dummy batch index

        min_index = np.argmin(input_sizes)

        in_channels = int(input_sizes[min_index])  # Assuming RGB images as input
        img_shape = [int(ins) for i,ins in enumerate(input_sizes) if i != min_index]
        
        # Create convolutional layers based on num_layers
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = num_channels  # Update in_channels for the next layer
        
        # Convert the list of layers to a Sequential container
        self.features = nn.Sequential(*layers)
        
        # Calculate the size of the input to the fully connected layers
        # Assuming the input image size is 32x32
        input_size = num_channels * int(np.prod(img_shape))
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)  # Assuming 10 output classes

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ImageClassificationModleeModel(modlee.modlee_model.ModleeModel):
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self(x)
        loss = self.loss_fn(y_out, y)
        return {'loss': loss}

    # def validation_step(self, val_batch, batch_idx):
    #     x, y = val_batch
    #     y_out = self(x)
    #     loss = self.loss_fn(y_out, y)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.001,
            momentum=0.9
        )
        return optimizer

from tensorflow import keras
from keras import layers, Input, Model, activations
import tensorflow.keras.applications as apps

from torch import nn
from torch.nn.modules.module import Module

import sys,io
from ast import literal_eval

from modlee_pypi.session import Info
default_info = Info()

import numpy as np
import math

def store_model_summary(model):
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    model.summary()

    sys.stdout = old_stdout
    
    return buffer.getvalue()


# def store_tuner_summary(tuner):
    
#     old_stdout = sys.stdout
#     sys.stdout = buffer = io.StringIO()

#     tuner.results_summary()

#     sys.stdout = old_stdout
    
#     return buffer.getvalue()



def get_model_performance_text(model,val_accuracy,info):
    
    key_val_seperator,parameter_seperator,layer_seperator = info.key_val_seperator,info.parameter_seperator,info.layer_seperator
    
    model_summary = store_model_summary(model)
    
    
    total_parameters = literal_eval(model_summary.split('Total params: ')[-1].split('\n')[0].replace(',',''))


    #shouldn't this be handled in prompt engineering or discretize?
    tp_base_2 = np.round(math.log(total_parameters)/math.log(2))
    total_parameters = int(2**tp_base_2)
    
    accuracy = max(val_accuracy)

    return "total_parameters{}{}{}accuracy{}{:.3f}".format(key_val_seperator,total_parameters,parameter_seperator,key_val_seperator,accuracy)



def get_model_details(model_layers_parameters):
    
    #fix to handle cases beyond classification in the future
        
    num_conv = len([l for l in [l for l in model_layers_parameters if 'layer_type' in l] if 'Conv' in l['layer_type']])
    num_dense = len([l for l in [l for l in model_layers_parameters if 'layer_type' in l] if 'Dense' in l['layer_type']])
    
    details = {'num_conv':num_conv,
               'num_dense':num_dense}
    
    return details

def get_parameter(layer,name):
    
    parameter = layer.__dict__[name]
    if type(parameter) != str:
        if name == 'activation':
            parameter = parameter.__name__
            
    return parameter

def convert_model_to_text(model,info: Info = default_info):
    '''
    Inputs
        model: the torch model to convert
    '''
    
    layer_seperator,parameter_seperator,key_val_seperator,info_parameters = info.layer_seperator,info.parameter_seperator,info.key_val_seperator,info.info_parameters

    #loop thru layers and and create text prompt

    layers_info = []
    
    wait_until_start_backend = False
    
    if 'base_model_info' in model.__dict__:
        base_model_layer_text = parameter_seperator.join(['{}{}{}'.format(key,key_val_seperator,value) for key,value in model.base_model_info.items()])
        layers_info.append(base_model_layer_text)
        wait_until_start_backend = True
    
    while model.__class__.__bases__[0] is not Module:
        model = model.children()
    # for layer in model.layers:
    for layer in model.children():
                
        layer_dir = layer.__dir__()
        
        layer_type = "layer_type{}{}".format(key_val_seperator,layer.__class__.__name__)
        # print('- {} -'.format(layer_type))

        # skip over the input layer
        if layer.__class__.__name__ != 'InputLayer':
                  
          layer_parameters = parameter_seperator.join(
              ['{}{}{}'.format(
                  p,key_val_seperator,get_parameter(layer,p)) for p in info_parameters.keys() if p in layer_dir])
          if len(layer_parameters)>0:
              layer_parameters = parameter_seperator+layer_parameters
          
          if wait_until_start_backend == False:
              layers_info.append(layer_type + layer_parameters)
          elif wait_until_start_backend == True and 'start_backend' in layer.__dict__:
              layers_info.append(layer_type + layer_parameters)
              wait_until_start_backend = False
        
    return layer_seperator.join(layers_info)


def process_model_parameters(p,type=None):

    # print(type)
    # print(p)

    if type == 'weights':
        if p == 'none':
            p = 'None'
    
    if type == 'include_top':
        try:
            p = int(literal_eval(p))
            if p == 0:
                p = 'False'
            else:
                p = 'True'
        except:
            if p == 'false':
                p = 'False'
            else:
                p = 'True'

    try:
        return literal_eval(p)
    except:
        return p


def convert_text_to_model(summary_text,prompt_text,info=default_info):
    # print('convert_text_to_model')
    
    layer_seperator,parameter_seperator,key_val_seperator = info.layer_seperator,info.parameter_seperator,info.key_val_seperator
    init_activation = info.init_activation
    unknown_value = info.unknown_value
        
    pretrained_model_names_dict = info.pretrained_model_names_dict

    init_activation = info.init_activation

    model_layers_parameters = []
    
    prompt_dict = { p.split(key_val_seperator)[0]:p.split(key_val_seperator)[1] for p in prompt_text.split(parameter_seperator) }
           

        
    model = keras.Sequential()
    base_model = None
    custom_inputs = None

    #? Need to handle case where summary contains a base model ...
    #? What about recreating results? training needs to happen in two steps for base model finetuning
    #? What about trained weights for best model? How do we load these weights? Maybe just save the model, and store in a dictionary with path as key and training data text as value
    print('summary_text: ',summary_text)
        
    layers_text = summary_text.split(layer_seperator)
    print('layers_text: ',layers_text)
    for i,lt in enumerate(layers_text):

        init_layers = {'SeparableConv2D'.lower():layers.SeparableConv2D(32,1),
                       'Conv2D'.lower():layers.SeparableConv2D(32,1),
                       'Dense'.lower():layers.Dense(256),
                       'Dropout'.lower():layers.Dropout(0.25),
                       'Flatten'.lower():layers.Flatten(),
                       'MaxPooling2D'.lower():layers.MaxPooling2D(1),
                       'GlobalAveragePooling2D'.lower():layers.GlobalAveragePooling2D(),
                       'BatchNormalization'.lower():layers.BatchNormalization(),
                       'Activation'.lower():layers.Activation(activation='linear'),
                      }
        
        print('lt: ',lt)
        layer_parameters = { s.split(key_val_seperator)[0]:s.split(key_val_seperator)[1] for s in lt.split(parameter_seperator) }
        print(layer_parameters)



        if 'model_type' in layer_parameters and i==0:
            #model_type = DenseNet121 - include_top = False - weights = imagenet - pooling = None - tp = 6953856 


            layer_parameters = { s.split(key_val_seperator)[0]:s.split(key_val_seperator)[1] for s in lt.split(parameter_seperator) }
            # layer_parameters['include_top'] = 
            # layer_parameters['weights'] = process_model_parameters(layer_parameters['weights'],type='weights')
            # print(layer_parameters)

            
            base_model = apps.__dict__[pretrained_model_names_dict[layer_parameters['model_type'].lower()]](include_top=process_model_parameters(layer_parameters['include_top'],type='include_top'), weights=process_model_parameters(layer_parameters['weights'],type='weights'), input_tensor=None, input_shape=process_model_parameters(prompt_dict['dataset_dims'])[1:], pooling=process_model_parameters(layer_parameters['pooling']), classes=process_model_parameters(prompt_dict['output_dim']))
            
            base_model_info = layer_parameters
            
            # if literal_eval(layer_parameters['include_top']) == True:
            if process_model_parameters(layer_parameters['include_top'],type='include_top') == True:
                #we know that base model is the full model
                inputs = base_model.input
                outputs = base_model.output
            else:
                inputs = base_model.input
                custom_inputs = base_model.output

                #freeze to start
                for b_layer in base_model.layers:
                    b_layer.trainable = False
            x = custom_inputs

        else:
            if i==0:            
                #force input to match the prompt
                custom_inputs = Input(shape=literal_eval(prompt_dict['dataset_dims'])[1:])
                x = custom_inputs



            layer = init_layers[layer_parameters['layer_type'].lower()]
            
#             print(layer)
            
            if base_model != None and i == 1:
                layer.start_backend = True
    #         print(layer)
            for key,val in layer_parameters.items():

                # our strategy now to deal with errors in summary is to ...
                #- lean on our initial layer settings
                #- skip vals == unknown_value
                # in the future we will want a better proccess like ...
                #- if conv based, pick last layer value, etc ... 

                if val != unknown_value:

                    try:
                        val = literal_eval(val)
                    except:
    #                     print('convert_text_to_model error')
                        pass

                    if key in layer.__dict__:
                        if key == 'activation':
                            layer.__dict__[key]=init_activation[val]
                        #----
                        elif key == 'units' and i==len(layers_text)-1:
                            layer.__dict__[key]=literal_eval(prompt_dict['output_dim'])
                            layer_parameters[key]=literal_eval(prompt_dict['output_dim'])
                        #----
                        else:
                            layer.__dict__[key]=val

#             model.add(layer)
            x = layer(x)
            
        model_layers_parameters.append(layer_parameters)
#         print(x)

        
        
    if base_model != None and custom_inputs == None:
#         print('base_model only')
        model = Model(inputs=inputs, outputs=outputs)
        model.base_model_info = base_model_info
    elif base_model != None and custom_inputs != None:
#         print('base_model with custom backend')
        model = Model(inputs=inputs, outputs=x)
        model.base_model_info = base_model_info
    else:
        #no base_model used
#         print('custom model')
        model = Model(inputs=custom_inputs, outputs=x)
        
        
    #? Note if i=0 we need to rename the first custom layer with 'start_backend'
        
        

    return model,model_layers_parameters





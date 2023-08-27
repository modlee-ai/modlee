from modlee import model_summary
# from model_summary import get_model_details
get_model_details = model_summary.get_model_details
from modlee import misc
# from utilities import discretize
discretize = misc.discretize

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,GlobalMaxPooling2D
import tensorflow.keras.applications as apps
from tensorflow import keras
from keras import layers, Input, Model, activations

import random
import numpy as np

#-------------------------------
# In the future have a build function for all supported use cases
#-------------------------------


def build_model(info):
        
    init_activation, pretrained_model_names = info.init_activation, info.pretrained_model_names
    
    model = info.model_a829dsakfndsakleowp
    model_layer_parameters = info.model_layers_parameters_a829dsakfndsakleowp
    
    input_dim = model.input.shape[1:]
    output_dim = model.output.shape[-1]#tweak to handle multi-dim output in future
    model_details = get_model_details(model_layer_parameters)
    
#     model = keras.Sequential()
#     model.add(Input(shape=input_dim))
        
    #------------------------------------------------
    #--- pretrained model hps
    pretrained_model = random.choice([True,True,False])#hp.Boolean('pretrained_model')
    include_top = random.choice([True,False,False,False])#hp.Boolean('include_top')#

    if pretrained_model == True:
#         pretrained_model_index = hp.Int("pretrained_model_index", 0, len(pretrained_model_names))
#         model_name = pretrained_model_names[pretrained_model_index]

        model_name = np.random.choice(pretrained_model_names)

        print('model_name: ',model_name)
    
        if include_top == True:
            weights = None
        else:
            weights = 'imagenet'#random.choice(['imagenet',None])  

#         pooling_options = [None,'avg','max']
#         pooling_index = hp.Int("pooling_index", 0, len(pooling_options))
#         pooling = pooling_options[pooling_index]

        pooling_options = [None,'avg','max']
        pooling_index = np.random.choice([l for l in range(0,len(pooling_options))])#hp.Int("pooling_index", 0, len(pooling_options))
        pooling = np.random.choice(pooling_options)

        # create the base pre-trained model
        base_model = apps.__dict__[model_name](include_top=include_top, weights=weights, input_tensor=None, input_shape=input_dim, pooling=pooling, classes=output_dim)

        tp = int(np.sum([np.prod(p.shape) for p in base_model.trainable_weights]))

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False
        
        base_model_info = {'model_type':str(model_name),'include_top':str(include_top),'weights':str(weights),'pooling':str(pooling),'tp':str(tp)}
        print('base_model_info: ',base_model_info)
    #------------------------------------------------
    #--- custom hps
    
    #---- hps
    conv_type = np.random.choice([l for l in range(0,1)])#hp.Int("conv_type", 0, 1)
    conv_type_dict = {0:'SeparableConv2D',1:'Conv2D'}
    conv_activation = np.random.choice(["relu", "elu", "selu"])
    conv_start_filters = np.random.choice([l for l in range(8,33,8)])#hp.Int("conv_start_filters", min_value=8, max_value=32, step=8)
    conv_filters_multiple = np.random.uniform(1,2)#hp.Float("conv_filters_multiple", min_value=1, max_value=2)
    conv_batchnorm = random.choice([True,False])#hp.Boolean("conv_batchnorm")
    conv_dropout = random.choice([True,False])#hp.Boolean("conv_dropout")
    conv_maxpool = random.choice([True,False])#hp.Boolean("conv_maxpool")
    
    dense_activation = random.choice(["relu", "elu", "selu"])#hp.Choice("dense_activation", ["relu", "elu", "selu"])
    dense_start_units = np.random.choice([l for l in range(32,257,32)])#hp.Int("dense_start_units", min_value=32, max_value=256, step=32)
    dense_units_multiple = np.random.uniform(0.5,1.0)#hp.Float("dense_units_multiple", min_value=0.5, max_value=1.0)
    dense_batchnorm = random.choice([True,False])#hp.Boolean("dense_batchnorm")
    dense_dropout = random.choice([True,False])#hp.Boolean("dense_dropout")
    
    
    #toggle paramters for pretrained models, if include_top == False, we need to set base model == non_trainable
    
    #? how to get summary with base model in single line?
    #? name first non-base layer with a specific identifier: say 'dense:start_backend'. 
    #----- We can easily build just the backend model using this layer identifier. 
    #------- then just grab the summary in the normal way and insert special base model line prior
    

    #? Set base model to not trainable .. later we will unfreeze full model and train further
    # ---- actually, to start we should also try just doing this in one shot ... either might work well. the init weights themselves will speed up training. And if classes are wildly different we might need to change the earlier layers

#     print('pretrained_model: ',pretrained_model)
#     print('include_top: ',include_top)


    if pretrained_model == True:
        
        if include_top == True:
            #we know that base model is the full model
            inputs = base_model.input
            outputs = base_model.output
        else:
            inputs = base_model.input
            custom_inputs = base_model.output
            x = custom_inputs
            
            print('x.shape: ',x.shape)
            #--------- add custom backend
            
            if pooling == 'avg' and len(x.shape)==5:
                reduce_layer = GlobalAveragePooling2D()
                reduce_layer.start_backend = True
            elif pooling == 'max' and len(x.shape)==5:
                reduce_layer = GlobalMaxPooling2D()
                reduce_layer.start_backend = True
            else:
                reduce_layer = layers.Flatten()#NOTE: in future we can do pooling here 
                reduce_layer.start_backend = True    

            x = reduce_layer(x)
            
            units = dense_start_units
            # Tune the number of layers.
#             for i in range(hp.Int("num_layers", max([1,int((model_details['num_dense']-1)*0.5)]), 3)):
            num_dense_layers = np.random.choice([l for l in range(max([1,int((model_details['num_dense']-1)*0.5)]), 3)])
            for i in range(num_dense_layers):

                x = layers.Dense(units=np.max([units,output_dim]),activation='linear')(x)

                if dense_batchnorm:
                    x = layers.BatchNormalization()(x)
                x = layers.Activation(activation=dense_activation)(x)
                if dense_dropout:
                    x = layers.Dropout(rate=0.25)(x)

                units = int(units*dense_units_multiple)

            x = layers.Dense(output_dim, activation="softmax")(x)
        
    else:
            
        #--- conv layers
        filters = conv_start_filters
        num_conv_layers = np.random.choice([l for l in range(max([1,int(model_details['num_conv']*0.5)]), int(model_details['num_conv']*3))])
        for i in range(num_conv_layers):

            init_layers = {'SeparableConv2D'.lower():layers.SeparableConv2D(32,1),
                           'Conv2D'.lower():layers.SeparableConv2D(32,1),
                          }
            
            if i==0:            
                #force input to match the prompt
                custom_inputs = Input(shape=input_dim)
                x = custom_inputs

            layer_kernel_size = np.random.choice([l for l in range(1,3,1)])#hp.Int(f"kernel_size_{i}", min_value=1, max_value=3, step=1)
            layer_stride = np.random.choice([l for l in range(1,2,1)])#hp.Int(f"strides_{i}", min_value=1, max_value=2, step=1)

            layer = init_layers[conv_type_dict[conv_type].lower()]
            layer.__dict__['filters']=discretize(filters)
            layer.__dict__['kernel_size']=discretize((layer_kernel_size,layer_kernel_size))
            layer.__dict__['strides']=discretize((layer_stride,layer_stride))
    #         layer.__dict__['activation']=init_activation[conv_activation]
            layer.__dict__['activation']=init_activation['linear']

            filters = int(filters * conv_filters_multiple)
#             model.add(layer)

            x = layer(x)

            if conv_batchnorm:
#                 model.add(layers.BatchNormalization())
                x = layers.BatchNormalization()(x)
#             model.add(layers.Activation(activation=conv_activation))
            x = layers.Activation(activation=conv_activation)(x)
            if conv_maxpool==True and input_dim[-2]//(2*(i+1))>5:
#                 model.add(layers.MaxPool2D(pool_size=(2, 2)))
                x = layers.MaxPool2D(pool_size=(2, 2))(x)
            if conv_dropout:
#                 model.add(layers.Dropout(rate=0.25))
                x = layers.Dropout(rate=0.25)(x)

#         model.add(layers.Flatten())
        x = layers.Flatten()(x)

        units = dense_start_units
        # Tune the number of layers.
        num_dense_layers = np.random.choice([l for l in range(max([1,int((model_details['num_dense']-1)*0.5)]), 3)])
#         for i in range(hp.Int("num_layers", max([1,int((model_details['num_dense']-1)*0.5)]), 3)):
        for i in range(num_dense_layers):

#             model.add(
#                 layers.Dense(
#                     # Tune number of units separately.
#                     units=np.max([discretize(units),output_dim]),#hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
#     #                 activation=hp.Choice("activation", ["relu", "tanh", "elu"]),
#                     activation='linear'
#                 )
#             )
            x = layers.Dense(units=np.max([discretize(units),output_dim]),activation='linear')(x)
    
            if dense_batchnorm:
#                 model.add(layers.BatchNormalization())
                x = layers.BatchNormalization()(x)
#             model.add(layers.Activation(activation=dense_activation))
            x = layers.Activation(activation=dense_activation)(x)
            if dense_dropout:
#                 model.add(layers.Dropout(rate=0.25))
                x = layers.Dropout(rate=0.25)(x)

            units = int(units*dense_units_multiple)

        #don't know how to quite capture this yet, add later
        # if hp.Boolean("dropout"):
        #     model.add(layers.Dropout(rate=0.25))
#         model.add(layers.Dense(output_dim, activation="softmax"))
        x = layers.Dense(output_dim, activation="softmax")(x)
            
    
        
    if pretrained_model == True and include_top == True:
#         print('base_model only')
        model = Model(inputs=inputs, outputs=outputs)
        model.base_model_info = base_model_info
    elif pretrained_model == True and include_top == False:
#         print('base_model with custom backend')
        model = Model(inputs=inputs, outputs=x)
        model.base_model_info = base_model_info
    else:
        #no base_model used
#         print('custom model')
        model = Model(inputs=custom_inputs, outputs=x)
        
    # learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



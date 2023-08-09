from tensorflow import keras
from keras import layers, Input, activations
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow.keras.applications as apps

import numpy as np

import random

class DevData:

  def __init__(self,info):

    self.info = info

    self.name = 'keras.datasets.cifar10'

    (x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x[:-10000]
    x_val = x[-10000:]
    y_train = y[:-10000]
    y_val = y[-10000:]


    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0


    self.x_train = x_train[:int(0.1*x_train.shape[0])]
    self.x_val = x_val[:int(0.1*x_val.shape[0])]
    self.x_test = x_test[:int(0.1*x_test.shape[0])]

    y_train = y_train[:int(0.1*y_train.shape[0])]
    y_val = y_val[:int(0.1*y_val.shape[0])]
    y_test = y_test[:int(0.1*y_test.shape[0])]

    output_dim = 10
    self.y_train = keras.utils.to_categorical(y_train, output_dim)
    self.y_val = keras.utils.to_categorical(y_val, output_dim)
    self.y_test = keras.utils.to_categorical(y_test, output_dim)
    self.output_dim = output_dim


    self.input_shape = x_train.shape[1:]

class DevModel:
  def __init__(self,info,input_shape,output_dim):

    self.info = info
    self.input_shape = input_shape
    self.output_dim = output_dim

  def basic_cnn(self):

    keras.backend.clear_session()

    model = keras.Sequential()
    model.add(Input(shape=self.input_shape))
    model.add(layers.SeparableConv2D(filters=64,kernel_size=1))
    model.add(layers.SeparableConv2D(filters=64,kernel_size=1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('linear'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=64,activation='relu',))
    model.add(layers.Dense(units=self.output_dim,activation='softmax',))

    return model

  def basic_pretrained_cnn(self):

    model_name = self.info.pretrained_model_names[0]

    include_top = random.choice([True,False])
    if include_top == True:
        weights = None
    else:
        weights = 'imagenet'#random.choice(['imagenet',None])    
    pooling = None#random.choice([None,'avg','max'])

    # create the base pre-trained model
    base_model = apps.__dict__[model_name](include_top=include_top, weights=weights, input_tensor=None, input_shape=self.input_shape, pooling=pooling, classes=self.output_dim)

    # print(include_top)
    # print(base_model.output_shape)

    #? Note: issue with tp not varying with include top
    tp = int(np.sum([np.prod(p.shape) for p in base_model.trainable_weights]))

    base_model_info = {'model_type':str(model_name),'include_top':str(include_top),'weights':str(weights),'pooling':str(pooling),'tp':str(tp)}


    # this is the model we will train
    if include_top == True:
        explore_model = base_model
    else:
        
        # add a global spatial average pooling layer
        x = base_model.output
        start_layer = GlobalAveragePooling2D()
        start_layer.start_backend = True
        x = start_layer(x)
        # let's add a fully-connected layer
        x = Dense(256, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.output_dim, activation='softmax')(x)
        
        explore_model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    explore_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    explore_model.base_model_info = base_model_info
    # if 'base_model_info' in explore_model.__dict__:

    # explore_model.summary()

    #? how to get summary with base model in single line?
    #? name first non-base layer with a specific identifier: say 'dense:start_backend'. 
    #----- We can easily build just the backend model using this layer identifier. 
    #------- then just grab the summary in the normal way and insert special base model line prior

    return explore_model


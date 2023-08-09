import tensorflow.keras.applications as apps
from keras import layers, Input, activations

class Info:

  def __init__(self):
    
    self.pretrained_model_names = [a for a in apps.__dir__() if a[0].isupper()]
    self.pretrained_model_names_dict = {a.lower():a for a in apps.__dir__() if a[0].isupper()}

    self.info_parameters = {'units':[],
                            'activation':[],
                            'filters':[],
                            'kernel_size':[],
                            'strides':[],
                            'padding':[],
                            'dilation_rate':[],
                            'rate':[],
                            'pool_size':[],
                            'depth_multiplier':[],
                            }

    self.init_activation = {'relu':activations.relu,
                            'elu':activations.elu,
                            'gelu':activations.gelu,
                            'selu':activations.selu,
                            'linear':activations.linear,
                            'softmax':activations.softmax,
                            'sigmoid':activations.sigmoid,
                            'tanh':activations.tanh,
                            }

    self.unknown_value = '[UNK]'
    self.layer_seperator = ' ~ '
    self.parameter_seperator = ' - '
    self.key_val_seperator = ' = '
    self.section_seperator = ' | '

def save_explore_details_to_global(model,model_layers_parameters,search_detail,info):
    info.model_a829dsakfndsakleowp = model
    info.model_layers_parameters_a829dsakfndsakleowp = model_layers_parameters
    info.search_detail_a829dsakfndsakleowp = search_detail


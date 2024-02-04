from .recommender import Recommender, RecommendedModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from torchvision.models import \
#     resnet34, ResNet34_Weights, \
#     resnet18, ResNet18_Weights, \
#     resnet152, ResNet152_Weights


import modlee
from modlee.converter import Converter
modlee_converter = Converter()

class ImageRecommender(Recommender):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.modality = 'image'
    
    def _append_classifier_to_model(self,model,num_classes):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.model_clf_layer = nn.Linear(1000, num_classes)

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

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
    
                
    def fit(self, dataloader, *args, **kwargs):
        super().fit(dataloader, *args, **kwargs)
        assert self.meta_features is not None
        # num_classes = len(dataloader.dataset.classes)
        if hasattr(dataloader.dataset, 'classes'):  
            num_classes = len(dataloader.dataset.classes)
        else:
            # try to get all unique values
            # assumes all classes will be represented in several batches
            unique_labels = set()
            n_samples = 0
            # while n_samples < 200:
            for d in dataloader.dataset:
                tgt = d[-1]
                # img,tgt = next(iter(dataloader))
                unique_labels.update(list(tgt.unique().cpu().numpy()))
                n_samples += len(tgt)
                # num_classes = len(tgt.unique())
            num_classes = len(unique_labels)
            # num_classes = 21
            # print(f'{unique_labels = }')
        self.meta_features.update({
            'num_classes': num_classes
        })
        # try:
        if 1:
            self.model_text = self._get_model_text(self.meta_features)
            # breakpoint()
            model = modlee_converter.onnx_text2torch(self.model_text)
            for param in model.parameters():
                # torch.nn.init.constant_(param,0.001)
                try:
                    torch.nn.init.xavier_normal_(param,1.0)
                except:
                    torch.nn.init.normal_(param)
            self.model = ImageRecommendedModel(model,loss_fn=self.loss_fn)

            self.code_text = self.get_code_text()
            self.model_code = modlee_converter.onnx_text2code(self.model_text)
            self.model_text = self.model_text.decode('utf-8')
            # breakpoint()
            clean_model_text = '>'.join(self.model_text.split('>')[1:])
            # typewriter_print(clean_model_onnx_text,sleep_time=0.005)
            # self.write_files()
            self.write_file(self.model_text, './modlee_model.txt')
            self.write_file(self.model_code, './modlee_model.py')
            
        # except:
        else:
            print("Could not retrieve model, could not access server or data features may be malformed.")
            self.model = None
    
    
class ImageClassificationRecommender(ImageRecommender):
    def __init__(self,
                #  dataloader, max_model_size_MB=10, num_classes=10,
                #  dataloader_input_inds=[0], min_accuracy=None, max_loss=None,
                 *args, **kwargs):

        # sleep(0.5)

        # print('---Contacting Modlee for a Recommended Image Classification Model--->\n')
        
        # sleep(0.5)
        # self.dataloader = dataloader
        # self.max_model_size_MB = max_model_size_MB
        # self.num_classes = num_classes
        # self.dataloader_input_inds = dataloader_input_inds
        # self.num_classes = num_classes
        
        super().__init__(*args, **kwargs)
        self.task = 'classification'
        self.loss_fn = F.cross_entropy
        
    # def recommend_model(self, meta_features):
    #     """
    #     Recommend a model based on meta-features

    #     Args:
    #         meta_features (_type_): A dictionary of meta-features

    #     Returns:
    #         torch.nn.Module: The recommended model
    #     """

    #     num_layers=1
    #     num_channels=8

    #     ret_model = VariableConvNet(num_layers,num_channels,self.input_sizes,self.num_classes)
    #     model_str = 'VariableConvNet({},{},{},{})'.format(num_layers,num_channels,self.input_sizes,self.num_classes)

    #     for i in range(10):

    #         model = VariableConvNet(int(num_layers),int(num_channels),self.input_sizes,self.num_classes)

    #         if get_model_size(model)<self.max_model_size_MB:
    #             ret_model = model
    #             num_layers += 1
    #             num_channels = num_channels*2
    #         else:
    #             break

    #     return ret_model,model_str


# class VariableConvNet(nn.Module):
#     def __init__(self, num_layers, num_channels, input_sizes_orig, num_classes):
#         super(VariableConvNet, self).__init__()

#         layers = []  # List to hold convolutional layers
        
#         input_sizes = input_sizes_orig[1:]#this is a dummy batch index

#         min_index = np.argmin(input_sizes)

#         in_channels = int(input_sizes[min_index])  # Assuming RGB images as input
#         img_shape = [int(ins) for i,ins in enumerate(input_sizes) if i != min_index]
        
#         # Create convolutional layers based on num_layers
#         for _ in range(num_layers):
#             layers.append(nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1))
#             layers.append(nn.ReLU())
#             in_channels = num_channels  # Update in_channels for the next layer
        
#         # Convert the list of layers to a Sequential container
#         self.features = nn.Sequential(*layers)
        
#         # Calculate the size of the input to the fully connected layers
#         # Assuming the input image size is 32x32
#         input_size = num_channels * int(np.prod(img_shape))
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(input_size, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, num_classes)  # Assuming 10 output classes

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class ImageSegmentationRecommender(ImageRecommender):
    def __init__(self, *args, **kwargs):

        # sleep(0.5)

        # print('---Contacting Modlee for a Recommended Image Classification Model--->\n')
        
        # sleep(0.5)
        # self.dataloader = dataloader
        # self.max_model_size_MB = max_model_size_MB
        # self.num_classes = num_classes
        # self.dataloader_input_inds = dataloader_input_inds
        # self.num_classes = num_classes
        
        super().__init__(*args, **kwargs)
        self.task = 'segmentation'
        # self.loss_fn = F.cross_entropy
        self.loss_fn = torch.nn.CrossEntropyLoss() 
        def squeeze_entropy_loss(x, *args, **kwargs):
            return torch.nn.CrossEntropyLoss()(x.squeeze)

class ImageRecommendedModel(RecommendedModel):
    def forward(self, x):
        # print(type(x))
        # x = torchvision.transforms.Resize((300,300))(x)
        return self.model(x)

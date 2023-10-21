import modlee

from datetime import datetime
import ssl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import lightning.pytorch as pl


def setup():

    # t1 = datetime.now()

    # !curl -o modlee-0.0.1.post6-py3-none-any.whl http://modlee.pythonanywhere.com/get_wheel/demos_demo01_modlee-0.0.1.post6-py3-none-any.whl -O
    # !curl -o modleesurvey-0.0.1-py3-none-any.whl http://modlee.pythonanywhere.com/get_wheel/demos_demo01_modleesurvey-0.0.1-py3-none-any.whl -O
    # !curl http://modlee.pythonanywhere.com/get_wheel/onnx2torch-1.5.11-py3-none-any.whl -O
    # !pip3 install -q 'modlee-0.0.1.post6-py3-none-any.whl' 'modleesurvey-0.0.1-py3-none-any.whl' 'onnx2torch-1.5.11-py3-none-any.whl' torchsummary==1.5.1 ipywidgets==7.7.1
    # !pip3 install -q onnx_graphsurgeon==0.3.27 --index-url https://pypi.ngc.nvidia.com

    import os


    demo_header = 'demos_demo01_'

    # os.system('curl -o modlee-0.0.1.post6-py3-none-any.whl http://modlee.pythonanywhere.com/get_wheel/{}modlee-0.0.1.post6-py3-none-any.whl -O'.format(demo_header))
    os.system('curl -o modleesurvey-0.0.1-py3-none-any.whl http://modlee.pythonanywhere.com/get_wheel/{}modleesurvey-0.0.1-py3-none-any.whl -O  > /dev/null 2>&1'.format(demo_header))
    os.system('curl http://modlee.pythonanywhere.com/get_wheel/onnx2torch-1.5.11-py3-none-any.whl -O  > /dev/null 2>&1')
    os.system("pip3 install -q 'modleesurvey-0.0.1-py3-none-any.whl' 'onnx2torch-1.5.11-py3-none-any.whl' torchsummary==1.5.1 ipywidgets==7.7.1  > /dev/null 2>&1")
    os.system("pip3 install -q onnx_graphsurgeon==0.3.27 --index-url https://pypi.ngc.nvidia.com  > /dev/null 2>&1")
    # os.system("rm -r ~/.cache/torch/hub/checkpoints/")
    ssl._create_default_https_context = ssl._create_unverified_context

    # t2 = datetime.now()
    # print("Time taken to download:", t2 - t1)

    import torch
    import torchvision.models as models

    # Load the pre-trained ResNet-18 model with weights
    resnet18 = models.resnet18(pretrained=True)
    # vgg16 = models.vgg16(pretrained=True)
    del resnet18#,vgg16


    # # Define URLs for the ResNet-18 and VGG-16 weights
    # resnet18_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    # vgg16_url = "https://download.pytorch.org/models/vgg16-397923af.pth"

    # # Define output file names
    # resnet18_filename = "resnet18_weights.pth"
    # vgg16_filename = "vgg16_weights.pth"

    # # Use wget to download the weights
    # os.system(f"wget -o {resnet18_url} -O {resnet18_filename}")
    # os.system(f"wget -o {vgg16_url} -O {vgg16_filename}")
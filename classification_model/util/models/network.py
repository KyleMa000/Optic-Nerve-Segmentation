# Author: Kyle Ma @ BCIL 
# Created: 09/23/2023
# Optic Nerve Classification Model

import torch
from torch.nn import init
from util.models.baseline.unet_binary import UnetBinary


def get_model(name, device):

    model =  globals()[name]()
    
    model.to(device)
    # test model
    model.eval()
    test_input = torch.rand(2,3,512,512).to(device)
    prediction = model(test_input.float())
    assert prediction.shape == (2,1), f"weird output shape: {prediction.shape}"

    # Weights initialization
    model.apply(_weights_init) 

    return model

    
def _weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

        init.xavier_normal_(m.weight.data, gain=0.02)

        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)





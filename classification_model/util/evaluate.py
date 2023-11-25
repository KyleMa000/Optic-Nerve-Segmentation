# Author: Kyle Ma @ BCIL 
# Created: 09/23/2023
# Optic Nerve Classification Model

import torch
import numpy as np
import torch.nn as nn
from util.data.augment import DataAug
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix



def evaluate(model, val_set, device):
    # start the evaluation mode
    model.eval()
    
    pos_weight = torch.FloatTensor([10.0])
    # Binary classification Loss
    scorer = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    running_loss = 0
    
    val_set = DataAug(val_set, False)


    # create test loader
    test_loader_args = dict(drop_last = False, shuffle = False, batch_size = 1, 
                num_workers = 0, pin_memory = True)
    test_loader = DataLoader(val_set, **test_loader_args)

    # store mask and prediction
    predictions = []
    labels = []

    # for every slices in the test loader
    for i, (images, sum, label) in enumerate(test_loader):

        # move the images to gpu or cpu
        images = images.to(device)

        # get our prediction
        with torch.no_grad():
            prediction = model(images.float())

            predictions.append(prediction.item())
            labels.append(label.item())

            bce_loss = scorer(prediction.flatten().cpu(), label.float())
            
            running_loss += bce_loss.item()
    

    
    predictions = np.array(predictions) > 0.5
    
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    model.train()

    
    return running_loss / len(val_set), tn, fp, fn, tp



            
            
    
    
    
    
    




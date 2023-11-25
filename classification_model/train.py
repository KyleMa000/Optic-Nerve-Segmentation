# Author: Kyle Ma @ BCIL 
# Created: 09/23/2023
# Optic Nerve Classification Model

import os
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Customized Files
from util.config import Config
from util.evaluate import evaluate
from util.data.augment import DataAug
from util.data.loader import load_nrrd
from util.models.network import get_model
from util.data.split import train_val_split



def run_model(model, device, dir_checkpoint, dir_exp):

    # 1. Read data from matlab file
    positive, negative = load_nrrd(config.training_directory)

    # 2. Train eval Split
    train_set, val_set = train_val_split(positive, negative, config.eval_ratio, config.fix_split)
    
    
    number_train = len(train_set)
    
    # perform data augmentation (elastic transform, horizontal flip)
    train_set = DataAug(train_set, True)

    # create data loader
    train_loader_args = dict(drop_last = True, shuffle = False, batch_size = config.batch_size, 
                       num_workers = 0, pin_memory = True)
    train_loader = DataLoader(train_set, **train_loader_args)

    # The original paper uses adam with learning rate of 0.001
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)
    
    pos_weight = torch.FloatTensor([10.0]).to(device)
    # Binary classification Loss
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # logging the training configuration
    logging.info(f'''Starting training:
        Model:           {model.__class__.__name__}
        Epochs:          {config.epoch_number}
        Batch size:      {config.batch_size}
        Learning rate:   {config.learning_rate}
        Training size:   {number_train}
    ''')


    # remember how many step total has been taken
    global_step = 0

    loss_list = []
    validation_loss_list = []

    tpr_list = []
    tnr_list = []
    
    max_tpr = 0
    max_tpr_model = None

    # train for 200 epochs
    for epoch in range(1, config.epoch_number+1):
        
        running_loss = 0
        
        # start the training
        model.train()

        with tqdm(total = number_train, desc = f'Epoch {epoch}/{config.epoch_number}', unit = ' img') as pbar:

            # for every batch of images
            for i, (images, sum, label) in enumerate(train_loader):

                # move the images to gpu or cpu
                images = images.to(device)
                label = label.to(device)
                
                # add zero grad
                optimizer.zero_grad()

                # get the prediction
                prediction = model(images.float())

                # calculate loss
                bce_loss = criterion(prediction.flatten(), label.float())
                
                # gradient descent
                bce_loss.backward()                
                optimizer.step()
                global_step += 1
                
                # update the tqdm pbar
                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': bce_loss.item()})
                
                # calculating average loss dice
                running_loss += bce_loss.item()


        # print loss
        logging.info('average train loss is {} ; step {} ; epoch {}.'.format(running_loss / len(train_loader), global_step, epoch))

        # get evaluation score
        val_loss, tn, fp, fn, tp = evaluate(model, val_set, device)



        # for plotting validation dice
        validation_loss_list.append(val_loss)
        # for plotting dice and loss
        loss_list.append(running_loss / len(train_loader))

        tpr_list.append(tp / (tp + fn))
        tnr_list.append(tn / (tn + fp))
        
        if tp / (tp + fn) > max_tpr:
            max_tpr_model = model.state_dict()
            max_tpr = tp / (tp + fn)
            
            logging.info(f'Max TPR Loss is {max_tpr}')


        # log the evaluation score
        logging.info(f'Validation Loss is {val_loss}')


    # save the model after job is done
    torch.save(model.state_dict(), os.path.join(dir_checkpoint,'COMPLETED.pt'))
    torch.save(max_tpr_model, os.path.join(dir_checkpoint,'max_tpr.pt'))


    logging.info('Training Completed Model Saved')



    x = np.linspace(1, len(loss_list), len(loss_list))

    plt.figure()
    plt.plot(x, loss_list, label="train", color="blue")
    plt.plot(x, validation_loss_list, label="test", color="orange")
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('{}/Loss.png'.format(dir_exp))
    plt.close()


    y = np.linspace(1, len(tnr_list), len(tnr_list))
    plt.figure()
    plt.plot(y, tpr_list, label="True Positive Rate", color="blue")
    plt.plot(y, tnr_list, label="True Negative Rate", color="red")


    plt.xlabel('Epochs')
    plt.ylabel('Ratio')
    plt.title('Confusion Matrix')
    plt.legend()
    plt.savefig('{}/ConfusionMatrix.png'.format(dir_exp))
    plt.close()


if __name__ == '__main__':

    config = Config()
    
    # create output directory
    dir_output = './outputs/'
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)
    
    # create output directory
    dir_exp = './outputs/{}'.format(config.exp_name)
    if not os.path.isdir(dir_exp):
        os.mkdir(dir_exp)
    

    # initialize the logging
    logging.basicConfig(filename='{}/Running.log'.format(dir_exp), level=logging.INFO, format='%(asctime)s: %(message)s')
    
    # use GPU if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log which device we are using
    logging.info(f"Model is Running on {device}.")


    model = get_model(config.model_type, device)



    # create check point directory
    dir_checkpoint = '{}/checkpoints/'.format(dir_exp)
    if not os.path.isdir(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    # run the model and save interrupt
    try:
        run_model(model, device, dir_checkpoint, dir_exp)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(dir_checkpoint,'INTERRUPTED.pt'))
        logging.info("Saved Interrupt at INTERRUPTED.pt")

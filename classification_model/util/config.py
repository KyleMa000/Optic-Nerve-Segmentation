# Author: Kyle Ma @ BCIL 
# Created: 05/12/2023
# Implementation of Automated Hematoma Segmentation


class Config():

    # Experiment Congigurations
    epoch_number = 80
    batch_size = 2
    learning_rate = 0.001
    eval_ratio = 0.2
    
    fix_split = True
    model_type = "UnetBinary"
    # UnetBinary


    # Data Loading Configurations
    training_directory = "#####"

    # Experiment Output Folder Name
    exp_series = '0927_on_classification'

    if fix_split:
        exp_name = exp_series +'_locksplit_' + model_type  + "_epoch" + str(epoch_number) + "_lr" + str(learning_rate)
    else:
        exp_name = exp_series + model_type + "_epoch" + str(epoch_number) + "_lr" + str(learning_rate)
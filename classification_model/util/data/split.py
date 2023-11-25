# Author: Kyle Ma @ BCIL 
# Created: 09/23/2023
# Optic Nerve Classification Model

import random
import logging
import numpy as np

def train_val_split(positive, negative, eval_ratio, fix_split):

    if fix_split:
        random.seed(727)
        

    num_eval_pos = int(eval_ratio * len(positive))
    num_eval_neg = int(eval_ratio * len(negative))

    train_set = []
    test_set = []

    test_set.extend(positive[:num_eval_pos])
    test_set.extend(negative[:num_eval_neg])

    train_set.extend(positive[num_eval_pos:])
    train_set.extend(negative[num_eval_neg:])

    random.shuffle(train_set)
    random.shuffle(test_set)

        
    logging.info(f"There are {len(train_set)} Training Data")
    logging.info(f"There are {len(test_set)} Validation Data")
        
    return train_set, test_set
# Author: Kyle Ma @ BCIL 
# Created: 09/23/2023
# Optic Nerve Classification Model

import os
import nrrd
import logging
import numpy as np
from tqdm import tqdm


def load_nrrd(directory):

    positive = []
    negative = []

    # log where are we loading the data from
    logging.info(f"Loading data from: {directory}")


    # all the patients are facing left
    for patient_path in tqdm(os.listdir(directory)):
    
        if patient_path.startswith('.'):
            continue

        

        # get paths where things are stored
        brain_image_path = os.path.join(directory, patient_path, 'img.nrrd')
        optic_left_path = os.path.join(directory, patient_path, 'structures/OpticNerve_L.nrrd')
        optic_right_path = os.path.join(directory, patient_path, 'structures/OpticNerve_R.nrrd')

        # load the nrrd images
        brain_image, _ = nrrd.read(brain_image_path)
        optic_left, _ = nrrd.read(optic_left_path)
        optic_right, _ = nrrd.read(optic_right_path)

        # get left and right optic nerves togehter
        optic = optic_left + optic_right

        # put them in the output format
        for i in range(brain_image.shape[-1]):

            if optic[:,:,i].any():
                positive.append([brain_image[:,:,i], int(np.sum(optic[:,:,i])), 1])
            else:
                negative.append([brain_image[:,:,i], 0, 0])


    # log how many data we loaded
    logging.info(f"Loaded {len(positive)} Slices of Positive Data")
    logging.info(f"Loaded {len(negative)} Slices of Negative Data")

    return positive, negative
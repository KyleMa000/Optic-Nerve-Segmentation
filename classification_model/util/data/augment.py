# Author: Kyle Ma @ BCIL 
# Created: 09/23/2023
# Optic Nerve Classification Model

import cv2
import random
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset as Dataset
from util.data.elastic_transform import do_elastic_transform

class DataAug(Dataset):
    def __init__(self, data, aug):
        self.data = data
        self.aug = aug

    def transform(self, dataset):
        
        temp_image = dataset[0].copy()
        
        # edge detection
        max_intensity = np.max(np.uint8(temp_image))

        lower_threshold = 0.4 * max_intensity
        upper_threshold = 1.2 * max_intensity

        image_blur = cv2.GaussianBlur(np.uint8(temp_image), (5, 5), 0)

        edges = cv2.Canny(image_blur, lower_threshold, upper_threshold)
        ###
        
        # histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        clahe_image = clahe.apply(np.uint8(temp_image))
        ###
        
        
        stacked_images = np.stack([temp_image, edges, clahe_image], axis=0)
                
        # Transform a copy to Tensor so original does not change
        image = TF.to_tensor(stacked_images).permute(1, 2, 0)
        sum = dataset[1]
        label = dataset[2]
        
        # Random Elastic Transformation
        
        if self.aug:
        
            if random.random() > 0.5:
    
                # do elastic transform to the image and mask at same time
                image = do_elastic_transform(image)
    
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
            
        sum = float(sum)
        label = float(label)
        
                    
        return image, sum, label

    def __getitem__(self, index):

        dataset = self.data[index]

        # apply the transformation
        image, sum, label = self.transform(dataset)


        return image, sum, label

    def __len__(self):
        return len(self.data)
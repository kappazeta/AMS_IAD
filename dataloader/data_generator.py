import os
import glob
import cv2
import argparse
import numpy as np 
import pandas as pd
import tensorflow as tf
import keras
import segmentation_models as sm

try:
    from keras.utils import Sequence
except:
    from keras.utils.all_utils import Sequence

class Dataset:
    CLASSES  = ['background', 'road']

    def __init__(self, images_fp, masks_fp, classes = None, augmentation = None, preprocessing = None):
        self.images_fp = images_fp
        self.masks_fp = masks_fp

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fp[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.masks_fp:
            mask = cv2.imread(self.masks_fp[i], 0)
            mask[mask == 1] = 0
            mask[mask == 255] = 1

            # extract certain classes from mask
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            if self.masks_fp:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                sample = self.preprocessing(image=image)
                image = sample['image']
        if self.masks_fp:
            return image, mask
        else:
            return image
    
    def __len__(self):
        return len(self.images_fp)

class DataLoader(Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return (batch[0], batch[1])

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
                         
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   

if __name__ == '__main__':
    #opt = parse_opt()
    main()

import os
import glob
import cv2
import argparse
import numpy as np 
import pandas as pd
from netCDF4 import Dataset as NetCDFDataset
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

class Dataset2:
    CLASSES  = ['background', 'road']

    def __init__(self, images_fp, features, classes=None, augmentation=None):
        self.images_fp = images_fp
        self.features = features

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation

    def __getitem__(self, i, cloud_threshold=1E-3):
        # Read NetCDF file.
        img_root = NetCDFDataset(self.images_fp[i], "r")

        # Load mask labels.
        mask = None
        if "Label" in img_root.variables:
            b_label = img_root["/Label"][:]
            b_label = np.where(b_label >= 0.5, 1, 0)
            masks = [(b_label == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float32')

        # Load features and stack them into a single tensor.
        bands = []
        for feature in self.features:
            b_image = img_root["/" + feature][:]
            b_image[b_image < 0] = 0
            b_image[b_image > 1] = 1

            bands.append(b_image)

        img_root.close()

        image = np.stack(bands, axis=-1).astype('float32')

        # Clear pixels on the mask which were cloudy on the image.
        mask_pre = None
        if mask is not None:
            mask_pre = mask.copy()
            mask_mask = image[:, :, 0] < cloud_threshold 
            mask[mask_mask] = 0

        # Perform data augmentation on the image.
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if mask is not None:
            return image, mask
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

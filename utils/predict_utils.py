import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import tensorflow as tf
import segmentation_models as sm
import keras
from dataloader.data_generator import Dataset, Dataset2, DataLoader
import utils.data_utils as du

sm.set_framework('tf.keras')
sm.framework()

ARCH_MAP = {'Unet' : sm.Unet}


def predict(config):
    DATA_DIR = config['data']['path']
    TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
    PRED_DATA_DIR = os.path.join(DATA_DIR, 'pred')

    ARCH = config['model']['architecture']
    BACKBONE = config['model']['backbone']
    BATCH_SIZE = config['train']['batch_size']

    CLASSES = ['background', 'road']
    LR = config['train']['learning_rate']
    EPOCHS = config['train']['epochs'] 
    LOSS = config['train']['loss']
    CHECKPOINTS_PATH = os.path.join('results', os.path.join(config['experiment_name'], 'Models'))
    MODEL_WEIGHTS = os.path.join(CHECKPOINTS_PATH, 'best_model.h5')
    N_CLASSES = config['model']['n_classes']

    if 'weights' in config['model']:
        MODEL_WEIGHTS = config['model']['weights']

    DATASTRUCT_VER = 1
    if 'datastruct_version' in config['data']:
        DATASTRUCT_VER = config['data']['datastruct_version']

    FEATURES = ['B02', 'B03', 'B04']
    if 'features' in config['data']:
        FEATURES = config['data']['features']

    if DATASTRUCT_VER == 1:
        ENCODER_WEIGHTS = 'imagenet'
        INPUT_SHAPE = None

        pred_images = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "Images/*")))[-200:]
        pred_masks = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "Masks/*")))[-200:]

        preprocess_input = sm.get_preprocessing(BACKBONE)

        pred_dataset = Dataset(pred_images, pred_masks, classes=CLASSES, preprocessing=du.get_preprocessing(preprocess_input))
        pred_dataloader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=True)
    elif DATASTRUCT_VER == 2:
        ENCODER_WEIGHTS = None
        INPUT_SHAPE = (None, None, len(FEATURES))

        pred_images = glob.glob(os.path.join(DATA_DIR, 'pred/*/*/*'))
        pred_dataset = Dataset2(pred_images, FEATURES, classes=CLASSES)
        pred_dataloader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        raise ValueError("Unsupported data structure version {}".format(DATASTRUCT_VER))

    # create model
    model = ARCH_MAP[ARCH](BACKBONE, classes=N_CLASSES, activation='softmax', encoder_weights=ENCODER_WEIGHTS, input_shape=INPUT_SHAPE)
    model.load_weights(MODEL_WEIGHTS) 
    
    # define optimizer
    optim = tf.keras.optimizers.Adam(LR)

    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 1.0]))
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5, class_weights=np.array([0.5,1.0]))]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, dice_loss, metrics)

    # Predict
    for i in range(0, len(pred_dataset), BATCH_SIZE):
        print("{} / {}".format(i, len(pred_dataset)))

        num_subtiles = min(BATCH_SIZE, len(pred_dataset) - i)
        images = [None]*num_subtiles
        for j in range(num_subtiles):
            images[j], _ = pred_dataset[i + j]
        images = np.stack(images)
        # Stack into batches.
        pr_masks = (model.predict(images).round() * 255).astype('uint8')

        # TODO:: Raw predictions into files.
    
        for j in range(num_subtiles):
            fpath = Path(pred_images[i + j])
            fname = fpath.stem
            product = fpath.parts[-3]
            os.makedirs(os.path.join("predictions", product), exist_ok=True)
            cv2.imwrite(os.path.join("predictions", product, fname + '.tif'), pr_masks[j, :, :, 1])


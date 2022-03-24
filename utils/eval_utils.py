import os
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


def evaluate(config):
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

        pred_images = glob.glob(os.path.join(DATA_DIR, 'test/*'))
        pred_dataset = Dataset2(pred_images, FEATURES, classes=CLASSES)
        pred_dataloader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=True)
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

    # Evaluate model results.
    scores = model.evaluate_generator(pred_dataloader)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))

    ### plot ###
    n = 10
    ids = np.random.choice(np.arange(len(pred_dataset)), size=n)
    ids = ids.tolist()
    
    for i in ids:
        image, gt_mask = pred_dataset[i]
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image).round()
        fig = du.visualize(image=du.denormalize(image.squeeze()), gt_mask=gt_mask[..., 0].squeeze(), pr_mask=pr_mask[..., 0].squeeze())
    
        fname = Path(pred_images[i]).stem
        plt.savefig('predicted_examples/pred_{}_{}.png'.format(i, fname), bbox_inches='tight')


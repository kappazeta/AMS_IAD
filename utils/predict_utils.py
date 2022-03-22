import os
import numpy as np
import matplotlib.pyplot as plt
import glob
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

    ARCH = config['model']['architecture']
    BACKBONE = config['model']['backbone']
    BATCH_SIZE = config['train']['batch_size']

    CLASSES = ['background', 'road']
    LR = config['train']['learning_rate']
    EPOCHS = config['train']['epochs'] 
    LOSS = config['train']['loss']
    CHECKPOINTS_PATH = os.path.join('results', os.path.join(config['experiment_name'], 'Models'))
    BEST_MODEL_PATH = os.path.join(CHECKPOINTS_PATH, 'best_model.h5')
    N_CLASSES = config['model']['n_classes']

    DATASTRUCT_VER = 1
    if 'datastruct_version' in config['data']:
        DATASTRUCT_VER = config['data']['datastruct_version']

    FEATURES = ['B02', 'B03', 'B04']
    if 'features' in config['data']:
        FEATURES = config['data']['features']

    preprocess_input = sm.get_preprocessing(BACKBONE)

    if DATASTRUCT_VER == 1:
        ENCODER_WEIGHTS = 'imagenet'
        INPUT_SHAPE = None

        test_images = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "Images/*")))[-200:]
        test_masks = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "Masks/*")))[-200:]

        test_dataset = Dataset(test_images, test_masks, classes=CLASSES, preprocessing=du.get_preprocessing(preprocess_input))
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    elif DATASTRUCT_VER == 2:
        ENCODER_WEIGHTS = None
        INPUT_SHAPE = (None, None, len(FEATURES))

        test_images = glob.glob(os.path.join(DATA_DIR, 'test/*'))
        test_dataset = Dataset2(test_images, FEATURES, classes=CLASSES, preprocessing=du.get_preprocessing(preprocess_input))
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        raise ValueError("Unsupported data structure version {}".format(DATASTRUCT_VER))

    # create model
    model = ARCH_MAP[ARCH](BACKBONE, classes=N_CLASSES, activation='softmax', encoder_weights=ENCODER_WEIGHTS, input_shape=INPUT_SHAPE)
    model.load_weights(BEST_MODEL_PATH) 
    # model.summary()
    
    # define optimizer
    optim = tf.keras.optimizers.Adam(LR)

    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 1.0]))
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5, class_weights=np.array([0.5,1.0]))]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, dice_loss, metrics)
    scores = model.evaluate_generator(test_dataloader)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))

    ### plot ###
    n = 5
    ids = np.random.choice(np.arange(len(test_dataset)), size=n)

    for i in ids:
        image, gt_mask = test_dataset[i]
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image).round()
        fig = du.visualize(image=du.denormalize(image.squeeze()), gt_mask=gt_mask[..., 0].squeeze(), pr_mask=pr_mask[..., 0].squeeze())
        plt.savefig('predicted_examples/predicted_segm_%d.png' % i, bbox_inches='tight')


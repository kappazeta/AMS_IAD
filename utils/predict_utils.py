import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import segmentation_models as sm
import keras
from dataloader.data_generator import Dataset, DataLoader
from utils.data_utils import get_training_augmentation, get_validation_augmentation, get_preprocessing, visualize, denormalize

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

    test_images = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "Images/*")))[-200:]
    test_masks = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "Masks/*")))[-200:]

    preprocess_input = sm.get_preprocessing(BACKBONE)

    #create model
    model = ARCH_MAP[ARCH](BACKBONE, classes=N_CLASSES, activation='softmax')
    model.load_weights(BEST_MODEL_PATH) 
    model.summary()
    
    test_dataset = Dataset(test_images, test_masks, classes = CLASSES, preprocessing = get_preprocessing(preprocess_input))
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # define optimizer
    optim = keras.optimizers.Adam(LR)

    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 1.0]))
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5, class_weights = np.array([0.5,1.0]))]

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
        fig = visualize(image=denormalize(image.squeeze()),gt_mask=gt_mask[..., 0].squeeze(),pr_mask=pr_mask[...,
        0].squeeze())
        plt.savefig('predicted_examples/predicted_segm_%d.png' % i, bbox_inches='tight')
        



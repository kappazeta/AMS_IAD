import os
import glob
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import keras
from dataloader.data_generator import Dataset, DataLoader
from utils.data_utils import get_training_augmentation, get_validation_augmentation, get_preprocessing

sm.set_framework('tf.keras')
sm.framework()

ARCH_MAP = {'Unet' : sm.Unet}

def define_callbacks(CHECKPOINTS_PATH):
    model_path =  os.path.join(CHECKPOINTS_PATH, 'best_model.h5')
    checkpointer = tf.keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True, save_best_only=True, mode='min')
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=70, monitor= 'val_f1-score', restore_best_weights = True, verbose = 1, mode = 'max', min_delta = 1E-7)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.1, mode = 'min', patience=25, verbose = 1)
    callbacks = [early_stopping, reduce_lr, checkpointer]
    return callbacks


def train(config):
    
    DATA_DIR = config['data']['path']
    VAL_SPLIT = 0.1

    ARCH = config['model']['architecture']
    BACKBONE = config['model']['backbone']
    N_CLASSES = config['model']['n_classes']

    BATCH_SIZE = config['train']['batch_size']
    CLASSES = ['background', 'road']
    LR = config['train']['learning_rate']
    EPOCHS = config['train']['epochs'] 
    LOSS = config['train']['loss']

    CHECKPOINTS_PATH = os.path.join('results', os.path.join(config['experiment_name'], 'Models'))

    images = glob.glob(os.path.join(DATA_DIR, 'Images/*'))
    masks = glob.glob(os.path.join(DATA_DIR, 'Masks/*'))
    val_indices = np.random.choice(range(len(images)), int(len(images) * VAL_SPLIT), replace = False)
    val_images = sorted(np.take(images, val_indices))
    val_masks = sorted(np.take(masks, val_indices))
    
    train_images = sorted(list(set(images) - set(val_images)))
    train_masks = sorted(list(set(masks) - set(val_masks)))

    preprocess_input = sm.get_preprocessing(BACKBONE)
    # define network parameters

    #create model
    model = ARCH_MAP[ARCH](BACKBONE, classes=N_CLASSES, activation='softmax', encoder_weights = 'imagenet')
    model.summary()
    
    # define optimizer
    opt = tf.keras.optimizers.Adam(LR)
    loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 1.0]))

    metrics = [sm.metrics.IOUScore(threshold=0.5),
    sm.metrics.FScore(threshold=0.5, class_weights = np.array([0.5, 1.0]))]
    model.compile(opt, loss, metrics)

    train_dataset = Dataset(train_images, train_masks, classes = CLASSES, augmentation = get_training_augmentation(), preprocessing = get_preprocessing(preprocess_input))
    valid_dataset = Dataset(val_images, val_masks, classes = CLASSES, augmentation = get_validation_augmentation(), preprocessing = get_preprocessing(preprocess_input))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)


    history = model.fit_generator(train_dataloader, steps_per_epoch=len(train_dataloader),epochs=EPOCHS, callbacks = define_callbacks(CHECKPOINTS_PATH), validation_data=valid_dataloader, validation_steps=len(valid_dataloader))


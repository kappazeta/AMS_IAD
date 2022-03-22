import os
import glob
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import keras
from dataloader.data_generator import Dataset, Dataset2, DataLoader
import utils.data_utils as du


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


def train(config, resume_training=False):
    DATA_DIR = config['data']['path']

    ARCH = config['model']['architecture']
    BACKBONE = config['model']['backbone']
    N_CLASSES = config['model']['n_classes']

    BATCH_SIZE = config['train']['batch_size']
    CLASSES = ['background', 'road']
    LR = config['train']['learning_rate']
    EPOCHS = config['train']['epochs'] 
    LOSS = config['train']['loss']

    CHECKPOINTS_PATH = os.path.join('results', os.path.join(config['experiment_name'], 'Models'))
    BEST_MODEL_PATH = os.path.join(CHECKPOINTS_PATH, 'best_model.h5')

    DATASTRUCT_VER = 1
    if 'datastruct_version' in config['data']:
        DATASTRUCT_VER = config['data']['datastruct_version']

    FEATURES = ['B02', 'B03', 'B04']
    if 'features' in config['data']:
        FEATURES = config['data']['features']

    DATA_AUGMENTATION = 1
    if 'augmentation' in config['data']:
        DATA_AUGMENTATION = config['data']['augmentation']

    preprocess_input = sm.get_preprocessing(BACKBONE)

    if DATASTRUCT_VER == 1:
        ENCODER_WEIGHTS = 'imagenet'
        INPUT_SHAPE = None

        VAL_SPLIT = 0.1
        images = glob.glob(os.path.join(DATA_DIR, 'Images/*'))
        masks = glob.glob(os.path.join(DATA_DIR, 'Masks/*'))
        val_indices = np.random.choice(range(len(images)), int(len(images) * VAL_SPLIT), replace=False)
        val_images = sorted(np.take(images, val_indices))
        val_masks = sorted(np.take(masks, val_indices))
        
        train_images = sorted(list(set(images) - set(val_images)))
        train_masks = sorted(list(set(masks) - set(val_masks)))

        train_dataset = Dataset(train_images, train_masks, classes=CLASSES, augmentation=du.get_training_augmentation(), preprocessing=du.get_preprocessing(preprocess_input))
        valid_dataset = Dataset(val_images, val_masks, classes=CLASSES, augmentation=du.get_validation_augmentation(), preprocessing=du.get_preprocessing(preprocess_input))

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    elif DATASTRUCT_VER == 2:
        ENCODER_WEIGHTS = None
        INPUT_SHAPE = (None, None, len(FEATURES))

        if DATA_AUGMENTATION == 1:
            train_aug = du.get_training_augmentation2()
        elif DATA_AUGMENTATION == 2:
            train_aug = du.get_training_augmentation3()
        else:
            raise ValueError("Unsupported data augmentation: {}".format(DATA_AUGMENTATION))

        train_images = glob.glob(os.path.join(DATA_DIR, 'train/*'))
        valid_images = glob.glob(os.path.join(DATA_DIR, 'val/*'))

        train_dataset = Dataset2(train_images, FEATURES, classes=CLASSES, augmentation=train_aug, preprocessing=du.get_preprocessing(preprocess_input))
        valid_dataset = Dataset2(valid_images, FEATURES, classes=CLASSES, augmentation=du.get_validation_augmentation2(), preprocessing=du.get_preprocessing(preprocess_input))

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    else:
        raise ValueError("Unsupported data structure version {}".format(DATASTRUCT_VER))

    # define network parameters

    # create model
    model = ARCH_MAP[ARCH](BACKBONE, classes=N_CLASSES, activation='softmax', encoder_weights=ENCODER_WEIGHTS, input_shape=INPUT_SHAPE)
    if resume_training:
        print("Loading weights from {}".format(BEST_MODEL_PATH))
        model.load_weights(BEST_MODEL_PATH) 
    else:
        model.summary()

    # define optimizer
    opt = tf.keras.optimizers.Adam(LR)
    loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 1.0]))

    metrics = [sm.metrics.IOUScore(threshold=0.5),
    sm.metrics.FScore(threshold=0.5, class_weights = np.array([0.5, 1.0]))]
    model.compile(opt, loss, metrics)

    history = model.fit_generator(train_dataloader, steps_per_epoch=len(train_dataloader),epochs=EPOCHS, callbacks=define_callbacks(CHECKPOINTS_PATH), validation_data=valid_dataloader, validation_steps=len(valid_dataloader))


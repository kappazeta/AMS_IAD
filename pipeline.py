import os
import glob
import cv2
import argparse
import numpy as np 
import pandas as pd
import tensorflow as tf
import keras
import segmentation_models as sm

# data utils import
from utils.utils import parse_config
from utils.train_utils import train
from utils.eval_utils import evaluate
from utils.predict_utils import predict

def parse_opt():
    parser = argparse.ArgumentParser(description = 'Run P2')
    parser.add_argument('--cfg', type = str, default = '', help = 'config.json path')
    parser.add_argument('--predict', dest='predict', action='store_true')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true')
    args = parser.parse_args()
    return args

def main(opt):
    
    config = parse_config(opt.cfg)

    if opt.train:
        train(config, resume_training=opt.resume)
    if opt.evaluate:
        evaluate(config)
    if opt.predict:
        predict(config)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

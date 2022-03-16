import pandas as pd 
import seaborn as sns
import tensorflow as tf
import numpy as np 
import os
import matplotlib.pyplot as plt
import imblearn
import json


def create_folders(cfg, config_path):
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    exp_dir = os.path.join(results_dir, cfg['experiment_name'])
    checkpoints_dir = os.path.join(exp_dir, 'Models')
    metadata_dir = os.path.join(exp_dir, 'meta_data')

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    cp_cfg_path = os.path.join(metadata_dir, 'config.json')
    if not os.path.exists(cp_cfg_path):
        os.popen('cp %s %s' % (config_path, cp_cfg_path))

def parse_config(config_path):
    cfg = json.load(open(config_path))
    create_folders(cfg, config_path)
    return cfg


import argparse
import cv2
import json
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# File path setup
current_dir = os.getcwd()
coco_dir = os.path.join(current_dir, 'datasets/coco/')
weights_dir = os.path.join(current_dir, 'trainedweights/')
os.makedirs(weights_dir, exist_ok=True)
pretrained_path = os.path.join(current_dir, 'control_sd21_ini.ckpt')
eval_results_dir = os.path.join(current_dir, 'eval_log')
os.makedirs(eval_results_dir, exist_ok=True)

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
model = create_model('./models/cldm_v21.yaml').to(device)
model.load_state_dict(torch.load(pretrained_path, map_location=device_name))
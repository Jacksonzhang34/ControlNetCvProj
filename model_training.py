import argparse
import cv2
import json
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Parse command-line arguments
def parse_args():
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser(description="Train a model on a custom dataset")
    parser.add_argument('--data_dir', type=str, default=os.path.join(current_dir, 'datasets/training/mini/'), help='Directory containing the data')
    parser.add_argument('--weights_dir', type=str, default=os.path.join(current_dir, 'models/trained weights/'), help='Directory to save the model weights')
    parser.add_argument('--resume_path', type=str, default=os.path.join(current_dir, 'models/control_sd21_ini.ckpt'), help='Path to resume model training')
    parser.add_argument('--checkpoint_name', type=str, default='mini_weights', help='Base name for the saved model checkpoint')
    return parser.parse_args()

args = parse_args()

# Dataset class
class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        with open(os.path.join(data_dir, 'prompt.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        source = cv2.imread(os.path.join(self.data_dir, source_filename))
        target = cv2.imread(os.path.join(self.data_dir, target_filename))
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0
        return dict(jpg=target, txt=prompt, hint=source)

# Callback for model checkpoint
checkpoint_callback = ModelCheckpoint(
    dirpath=args.weights_dir,
    filename=args.checkpoint_name,
    save_top_k=1,
    verbose=True
)

# Model preparation (assuming create_model function exists and works as expected)
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
model.learning_rate = 1e-5  # Set learning rate here if not using argparse for this

# Dataset and DataLoader
dataset = MyDataset(args.data_dir)
dataloader = DataLoader(dataset, num_workers=0, batch_size=4, shuffle=True)  # Set batch size here if not using argparse for this

# Trainer configuration
trainer = pl.Trainer(accelerator='gpu', precision=16, callbacks=[checkpoint_callback], max_epochs=10)
trainer.fit(model, dataloader)

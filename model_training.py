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
    parser.add_argument('--data_dir', type=str, default=os.path.join(current_dir, 'datasets/coco/'), help='Directory containing the data')
    parser.add_argument('--weights_dir', type=str, default=os.path.join(current_dir, 'trainedweights/'), help='Directory to save the model weights')
    parser.add_argument('--resume_path', type=str, default=os.path.join(current_dir, 'control_sd21_ini.ckpt'), help='Path to resume model training')
    parser.add_argument('--checkpoint_name', type=str, default='trained_weights', help='Base name for the saved model checkpoint') #named the weights to be the type of promt.json file used
    parser.add_argument('--prompt', type=str, default='prompt.json', help='caption type')
    return parser.parse_args()

args = parse_args()

# Dataset class
class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        with open(os.path.join(data_dir, args.prompt), 'rt') as f:
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
    
    def subset(self, start, end):
        return MyDataset(self.data_dir, self.data[start:end])
    

def train_val_test_split(data, train_ratio, val_ratio):
  n = len(data)
  train_end = int(n * train_ratio)
  val_end = int(n * (train_ratio + val_ratio))

  train_data = data.subset(0, train_end)
  val_data = data.subset(train_end, val_end)
  test_data = data.subset(val_end, n)

  return train_data, val_data, test_data

# Callback for model checkpoint
checkpoint_callback = ModelCheckpoint(
    dirpath=args.weights_dir,
    filename=args.checkpoint_name,
    save_top_k=1,
    verbose=True
)


resume_path = args.resume_path
batch_size = 4 # todo set
logger_freq = 300 # todo set
learning_rate = 1e-5 # todo set
sd_locked = True
only_mid_control = False


model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Dataset and DataLoader
full_dataset = MyDataset(args.data_dir)

# Splitting data
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
train_dataset, val_dataset, test_dataset = train_val_test_split(full_dataset, train_ratio, val_ratio)

train_loader = DataLoader(train_dataset, num_workers=2, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, num_workers=2, batch_size=batch_size, shuffle=False) # todo set num_workers
test_loader = DataLoader(test_dataset, num_workers=2, batch_size=batch_size, shuffle=False)

# Trainer configuration
trainer = pl.Trainer(accelerator='gpu', precision=16, callbacks=[checkpoint_callback], max_epochs=10)
trainer.fit(model,  train_loader, val_loader)

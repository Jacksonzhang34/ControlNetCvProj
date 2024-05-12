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

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a model on a custom dataset")
    parser.add_argument('--prompt', type=str, help='Caption type', required=True)
    return parser.parse_args()

args = parse_args()

# File path setup
current_dir = os.getcwd()
coco_dir = os.path.join(current_dir, 'datasets/coco/')
weights_dir = os.path.join(current_dir, 'trainedweights/')
pretrained_path = os.path.join(current_dir, 'control_sd21_ini.ckpt')
eval_results_dir = os.path.join(current_dir, 'eval_results')
os.makedirs(args.eval_results_dir, exist_ok=True)

# Data setup
class MyDataset(Dataset):   
    def __init__(self, data_dir, subset=None):
        self.data_dir = data_dir
        if subset is not None:
            self.data = subset
        else:
            self.data = []
            with open(os.path.join(self.data_dir, 'prompts', args.prompt), 'rt') as f:
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


# Configs
batch_size = 4 # todo set hyperparameter
logger_freq = 300 # todo set hyperparameter
learning_rate = 1e-5 # todo set hyperparameter
sd_locked = True
only_mid_control = False

model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

full_dataset = MyDataset(coco_dir)
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
train_dataset, val_dataset, test_dataset = train_val_test_split(full_dataset, train_ratio, val_ratio)
train_loader = DataLoader(train_dataset, num_workers=2, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, num_workers=2, batch_size=batch_size, shuffle=False) # todo set num_workers
test_loader = DataLoader(test_dataset, num_workers=2, batch_size=batch_size, shuffle=False)

# Training setup
image_logger = ImageLogger(batch_frequency=logger_freq)
tb_logger = TensorBoardLogger(save_dir="tb_log", name="ControlNet")
checkpoint_callback = ModelCheckpoint(
    dirpath=weights_dir,
    filename=os.path.splitext(args.prompt)[0],
    save_top_k=1,
    verbose=True
)
trainer = pl.Trainer(
    accelerator='gpu',
    precision=16,
    callbacks=[checkpoint_callback],
    max_epochs=10,
    logger=[tb_logger, image_logger]
)

# Training
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Evalution
eval_results = trainer.test(model=model, dataloaders=test_loader, ckpt_path="best")
result_path = os.path.join(eval_results_dir, f"{os.path.splitext(args.prompt)[0]}_results.json")
with open(result_path, "w") as f:
    json.dump(eval_results, f)
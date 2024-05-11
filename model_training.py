import torch
import sys
import os
import torchvision
import json
import cv2
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('/content/drive/My Drive/CV-JV-final-project/datasets/training/mini/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('/content/drive/My Drive/CV-JV-final-project/datasets/training/mini/' + source_filename)
        target = cv2.imread('/content/drive/My Drive/CV-JV-final-project/datasets/training/mini/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    

checkpoint_callback = ModelCheckpoint(
    dirpath='/content/drive/My Drive/CV-JV-final-project/models/trained weights/',  # Directory to save the model
    filename='mini_weights',
    save_top_k=1,
    verbose=True
)


resume_path = '/content/drive/My Drive/CV-JV-final-project/models/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
# model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.load_state_dict(torch.load(resume_path, map_location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)


trainer = pl.Trainer(accelerator='gpu', precision=16, callbacks=[logger, checkpoint_callback], max_epochs=10)
trainer.fit(model, dataloader)



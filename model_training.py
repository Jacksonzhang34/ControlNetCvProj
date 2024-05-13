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
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

print('1')
# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a model on a custom dataset"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Caption type")
    return parser.parse_args()


args = parse_args()
print('2')
# File path setup
current_dir = os.getcwd()
coco_dir = os.path.join(current_dir, "datasets/coco/")  # training data directory
weights_dir = os.path.join(
    current_dir, "trainedweights/"
)  # location of trained weights to be saved
os.makedirs(weights_dir, exist_ok=True)
pretrained_path = os.path.join(
    current_dir, "control_sd21_ini.ckpt"
)  # pretrained weights
# eval_results_dir = os.path.join(current_dir, 'eval_log')
# os.makedirs(eval_results_dir, exist_ok=True)
print('3')
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print('4')

# Data setup
class MyDataset(Dataset):
    def __init__(self, data_dir, subset=None):
        self.data_dir = data_dir
        if subset is not None:
            self.data = subset
        else:
            self.data = []
            with open(
                os.path.join(self.data_dir, "prompts", f"{args.prompt}.json"), "rt"
            ) as f:
                for line in f:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item["prompt"]

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
batch_size = 4  # can fine-tune
logger_freq = 300  # can fine-tune
learning_rate = 1e-5  # can fine-tune
num_workers = 2  # can fine-tune
sd_locked = True
only_mid_control = False

print('5')
model = create_model("./models/cldm_v21.yaml").to(device)
model.load_state_dict(torch.load(pretrained_path, map_location=device_name))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
print('6')
full_dataset = MyDataset(coco_dir)
train_ratio, val_ratio, test_ratio = 0.4, 0.1, 0.5
train_dataset, val_dataset, test_dataset = train_val_test_split(
    full_dataset, train_ratio, val_ratio
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
print('7')
# Training setup
image_logger = ImageLogger(batch_frequency=logger_freq)
tb_logger = TensorBoardLogger(save_dir="tb_log", name="ControlNet")
csv_logger = CSVLogger("csv_log", name="ControlNet")
# checkpoint_callback = ModelCheckpoint(
#     dirpath=weights_dir,
#     filename=args.prompt,
#     save_top_k=1,
#     verbose=True
# )
print('8')
trainer = pl.Trainer(
    accelerator="gpu", precision=16, max_epochs=10, logger=[tb_logger, csv_logger, image_logger]
)
print('9')
# trainer = pl.Trainer(accelerator="gpu", precision=16, max_epochs=10, logger=[tb_logger])

# Training
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print('10')
torch.save(model.state_dict(), os.path.join(weights_dir, args.prompt))
print('11')

# Evaluation
# eval_results = trainer.test(model=model, dataloaders=test_loader, ckpt_path="best")
# result_path = os.path.join(eval_results_dir, f"{args.prompt}_results.json")
# with open(result_path, "w") as f:
#     json.dump(eval_results, f)

import torch
import sys
import os
import torchvision
import json
import cv2
import numpy as np
from torch.utils.data import Dataset

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch

from share import *
from cldm.model import create_model, load_state_dict
import cv2
from annotator.util import resize_image
import numpy as np
import einops
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor



resume_path = '/content/drive/My Drive/CV-JV-final-project/models/trained weights/mini_weights.ckpt' # your checkpoint path
N = 1
ddim_steps = 30

model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

sample_path = '/content/drive/My Drive/CV-JV-final-project/datasets/training/mini/source/0.png'

def process(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = resize_image(img, 512)

  control = torch.from_numpy(img.copy()).float().cuda() / 255.0
  control = torch.stack([control for _ in range(N)], dim=0)
  control = einops.rearrange(control, 'b h w c -> b c h w').clone()
  c_cat = control.cuda()
  c = model.get_unconditional_conditioning(N)
  uc_cross = model.get_unconditional_conditioning(N)
  uc_cat = c_cat
  uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
  cond={"c_concat": [c_cat], "c_crossattn": [c]}
  b, c, h, w = cond["c_concat"][0].shape
  shape = (4, h // 8, w // 8)

  samples, intermediates = ddim_sampler.sample(ddim_steps, N,
                                              shape, cond, verbose=False, eta=0.0,
                                              unconditional_guidance_scale=9.0,
                                              unconditional_conditioning=uc_full
                                              )
  x_samples = model.decode_first_stage(samples)
  x_samples = x_samples.squeeze(0)
  x_samples = (x_samples + 1.0) / 2.0
  x_samples = x_samples.transpose(0, 1).transpose(1, 2)
  x_samples = x_samples.cpu().numpy()
  x_samples = (x_samples * 255).astype(np.uint8)

  return x_samples


def process_image_pair(img_path, label_path):
    predicted = process(img_path)
    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    # label = cv2.resize(label, (predicted.shape[1], predicted.shape[0]))
    mse = np.mean((predicted - label) ** 2)
    ssim_value = ssim(predicted, label, multichannel=True)
    return mse, ssim_value

def evaluate(input_dir, label_dir, num_images):
    input_paths = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))][:num_images]
    label_paths = [os.path.join(label_dir, f) for f in sorted(os.listdir(label_dir))][:num_images]

    mse_scores = []
    ssim_scores = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(process_image_pair, input_paths, label_paths)
        for mse, ssim_value in results:
            mse_scores.append(mse)
            ssim_scores.append(ssim_value)

    avg_mse = np.mean(mse_scores)
    avg_ssim = np.mean(ssim_scores)

    print(f"Average MSE: {avg_mse}")
    print(f"Average SSIM: {avg_ssim}")

    return avg_mse, avg_ssim


source = '/content/drive/My Drive/CV-JV-final-project/datasets/training/mini/source/'
target = '/content/drive/My Drive/CV-JV-final-project/datasets/training/mini/target/'

eval = evaluate(source, target, 10)
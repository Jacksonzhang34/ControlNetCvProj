import torch
import sys
import os
import torchvision
import json
import cv2
import numpy as np
import argparse
import random
import gradio as gr
from torch.utils.data import Dataset

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch

from share import *
from cldm.model import create_model, load_state_dict
from annotator.util import resize_image, HWC3
import einops
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
from skimage.metrics import structural_similarity
from image_similarity_measures.quality_metrics import fsim, ssim
from concurrent.futures import ThreadPoolExecutor

# from dataset import *
# sys.path.append("..")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a model on a custom dataset"
    )
    parser.add_argument("--weightName", type=str, required=True, help="Caption type")
    return parser.parse_args()

args = parse_args()

current_dir = os.getcwd()
pretrained_path = os.path.join(current_dir, "control_sd21_ini.ckpt") 
coco_dir = os.path.join(current_dir, "datasets/coco/")
weights_dir = os.path.join(current_dir, "trainedweights/")
eval_dir = os.path.join(current_dir, "eval/")
eval_logs_dir = os.path.join(eval_dir, "logs/")
eval_metrics_dir = os.path.join(eval_dir, "metrics/")
eval_img_pred_weight_dir = os.path.join(eval_dir, "images/predicted/", args.weightName)
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(eval_logs_dir, exist_ok=True)
os.makedirs(eval_metrics_dir, exist_ok=True)
os.makedirs(eval_img_pred_weight_dir, exist_ok=True)

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.4, 0.1, 0.1
A_PROMPT_DEFAULT = "best quality, extremely detailed"
N_PROMPT_DEFAULT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"


def get_data_paths(data_dir):
    data = []
    with open(
        os.path.join(data_dir, "prompts", f"{args.weightName}.json"), "rt"
    ) as f:
        for line in f:
            data.append(json.loads(line))

    n = len(data)
    test_start = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_num = int(n * TEST_RATIO)
    
    img_paths = [os.path.join(data_dir, data[i]["source"]) for i in range(test_start, test_start + test_num)]
    label_paths = [os.path.join(data_dir, data[i]["target"]) for i in range(test_start, test_start + test_num)]
    captions = [data[i]["prompt"] for i in range(test_start, test_start + test_num)]
    
    return img_paths, label_paths, captions


def run_sampler(
    model,
    input_image: np.ndarray,
    prompt: str,
    num_samples: int = 1,
    image_resolution: int = 512,
    seed: int = -1,
    a_prompt: str = A_PROMPT_DEFAULT,
    n_prompt: str = N_PROMPT_DEFAULT,
    guess_mode=False,
    strength=1.0,
    ddim_steps=50,
    eta=0.0,
    scale=9.0,
    show_progress: bool = True,
):
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()

        ddim_sampler = DDIMSampler(model)

        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        pl.seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            show_progress=show_progress,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]

        return np.asarray(results[0])


def process_image_pair(img_path, label_path, caption):
    print(img_path, label_path, caption)
    # predicted = process(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_image(img, 512)
    predicted = run_sampler(model, img, caption)
    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    # label = cv2.resize(label, (predicted.shape[1], predicted.shape[0]))
    mse = np.mean((predicted - label) ** 2)
    ssim1_value = structural_similarity(predicted, label, channel_axis = -1)
    ssim_value = ssim(predicted, label)
    fsim_value = fsim(predicted, label)
    
    return predicted, mse, ssim1_value, ssim_value, fsim_value


def evaluate(img_paths, label_paths, captions):

    mse_scores, ssim1_scores, ssim_scores, fsim_scores = [], [], [], []
    for i, (img_path, label_path, caption) in enumerate(zip(img_paths, label_paths, captions)):
        predicted, mse, ssim1_value, ssim_value, fsim_value = process_image_pair(img_path, label_path, caption)
        mse_scores.append(mse)
        ssim1_scores.append(ssim1_value)
        ssim_scores.append(ssim_value)
        fsim_scores.append(fsim_value)

        if i % 50 == 0:
            image_save_path = os.path.join(eval_img_pred_weight_dir, f"{os.path.basename(img_path)}")
            cv2.imwrite(image_save_path, cv2.cvtColor(predicted, cv2.COLOR_RGB2BGR))
            eval_logs_file_path = os.path.join(eval_logs_dir, f'{args.weightName}.txt')
            with open(eval_logs_file_path, "a") as log_file:
                log_file.write(f"{i}: {os.path.basename(img_path)}, {os.path.basename(label_path)}\n")
            
            avg_mse = np.mean(mse_scores)
            avg_ssim1 = np.mean(ssim1_scores)
            avg_ssim = np.mean(ssim_scores)
            avg_fsim = np.mean(fsim_scores)

            eval_metrics_file_path = os.path.join(eval_metrics_dir, f'{args.weightName}.txt')
            with open(eval_metrics_file_path, 'a') as file:
                file.write(f"i: {i}\n")
                file.write(f"Average MSE: {avg_mse}\n")
                file.write(f"Average SSIM1: {avg_ssim1}\n")
                file.write(f"Average SSIM: {avg_ssim}\n")
                file.write(f"Average FSIM: {avg_fsim}\n\n")

    avg_mse = np.mean(mse_scores)
    avg_ssim1 = np.mean(ssim1_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_fsim = np.mean(fsim_scores)

    eval_metrics_file_path = os.path.join(eval_metrics_dir, f'{args.weightName}.txt')
    with open(eval_metrics_file_path, 'a') as file:
        file.write(f" ----- OVERALL ----- ")
        file.write(f"Average MSE: {avg_mse}\n")
        file.write(f"Average SSIM1: {avg_ssim1}\n")
        file.write(f"Average SSIM: {avg_ssim}\n")
        file.write(f"Average FSIM: {avg_fsim}\n")

    return avg_mse, avg_ssim1, avg_ssim, avg_fsim


device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(device_name)
device = torch.device(device_name)
model = create_model('./models/cldm_v21.yaml').to(device)
weights_path = os.path.join(weights_dir, args.weightName)
model.load_state_dict(torch.load(weights_path))
print("model loaded")

img_paths, label_paths, captions = get_data_paths(coco_dir)
avgs = evaluate(img_paths, label_paths, captions)
print(f"Metric Averages: {avgs}")
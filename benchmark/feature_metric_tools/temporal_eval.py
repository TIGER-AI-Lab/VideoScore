import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.models import vit_b_16 
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage import io, color

ROUND_DIGIT=3
NUM_ASPECT=5

CLIP_POINT_HIGH=0.97
CLIP_POINT_MID=0.9
CLIP_POINT_LOW=0.8

DINO_POINT_HIGH=0.95
DINO_POINT_MID=0.85
DINO_POINT_LOW=0.75

TEM_SSIM_POINT_HIGH=0.9
TEM_SSIM_POINT_MID=0.75
TEM_SSIM_POINT_LOW=0.6


def clip_inter_frame(model,tokenizer,frames_path_list):
    device=model.device
    frame_sim_list=[]
    for f_idx in range(len(frames_path_list)-1):
        frame_1 = Image.open(frames_path_list[f_idx])
        frame_2 = Image.open(frames_path_list[f_idx+1])
        input_1 = tokenizer(images=frame_1, return_tensors="pt", padding=True).to(device)
        input_2 = tokenizer(images=frame_2, return_tensors="pt", padding=True).to(device)
        output_1 = model.get_image_features(**input_1).flatten()
        output_2 = model.get_image_features(**input_2).flatten()
        cos_sim = F.cosine_similarity(output_1, output_2, dim=0).item()
        frame_sim_list.append(cos_sim)
    clip_frame_score = np.mean(frame_sim_list)
    ans=[]
    if clip_frame_score >= CLIP_POINT_HIGH:
        ans=[4]*NUM_ASPECT
    elif clip_frame_score < CLIP_POINT_HIGH and clip_frame_score >= CLIP_POINT_MID:
        ans=[3]*NUM_ASPECT
    elif clip_frame_score < CLIP_POINT_MID and clip_frame_score >= CLIP_POINT_LOW:
        ans=[2]*NUM_ASPECT
    else:
        ans=[1]*NUM_ASPECT
    return clip_frame_score, ans


def dino_inter_frame(model, preprocess, frames_path_list):
    device=model.device
    frame_sim_list=[]
    for f_idx in tqdm(range(len(frames_path_list)-1)):
        frame_1=Image.open(frames_path_list[f_idx])
        frame_2=Image.open(frames_path_list[f_idx+1])
        frame_tensor_1 = preprocess(frame_1).unsqueeze(0).to(device)
        frame_tensor_2 = preprocess(frame_2).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_1 = model(frame_tensor_1).flatten()
            feat_2 = model(frame_tensor_2).flatten()
        cos_sim=F.cosine_similarity(feat_1, feat_2, dim=0).item()
        frame_sim_list.append(cos_sim)
    frame_sim_avg=np.mean(frame_sim_list)
    ans=[0 for _ in range(NUM_ASPECT)]
    if frame_sim_avg >= DINO_POINT_HIGH:
        ans=[4]*NUM_ASPECT
    elif frame_sim_avg < DINO_POINT_HIGH and frame_sim_avg >= DINO_POINT_MID:
        ans=[3]*NUM_ASPECT
    elif frame_sim_avg < DINO_POINT_MID and frame_sim_avg >= DINO_POINT_LOW:
        ans=[2]*NUM_ASPECT
    else:
        ans=[1]*NUM_ASPECT
    return frame_sim_avg, ans


def ssim_inter_frame(frames_path_list):
    ssim_list=[]
    for f_idx in range(len(frames_path_list)-1):
        frame_1=Image.open(frames_path_list[f_idx])
        frame_1_gray=color.rgb2gray(frame_1)
        frame_2=Image.open(frames_path_list[f_idx+1])
        frame_2_gray=color.rgb2gray(frame_2)

        ssim_value, _ = ssim(frame_1_gray, frame_2_gray, full=True,\
                                 data_range=frame_2_gray.max() - frame_2_gray.min())
        ssim_list.append(ssim_value)
    ssim_avg=np.mean(ssim_list)
    ans=[]
    if ssim_avg >= TEM_SSIM_POINT_HIGH:
        ans=[4]*NUM_ASPECT
    elif ssim_avg < TEM_SSIM_POINT_HIGH and ssim_avg >= TEM_SSIM_POINT_MID:
        ans=[3]*NUM_ASPECT
    elif ssim_avg < TEM_SSIM_POINT_MID and ssim_avg >= TEM_SSIM_POINT_LOW:
        ans=[2]*NUM_ASPECT
    else:
        ans=[1]*NUM_ASPECT
    return ssim_avg, ans

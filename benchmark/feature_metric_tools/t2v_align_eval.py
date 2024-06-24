import numpy as np
from PIL import Image
import torch.nn.functional as F
from typing import List

NUM_ASPECT=5
ROUND_DIGIT=3
MAX_LENGTH = 76

MAX_NUM_FRAMES=8

CLIP_POINT_LOW=0.27
CLIP_POINT_MID=0.31
CLIP_POINT_HIGH=0.35

X_CLIP_POINT_LOW=0.15
X_CLIP_POINT_MID=0.225
X_CLIP_POINT_HIGH=0.30


def clip_score(
    model, 
    tokenizer,
    text:str,
    frame_path_list:List[str],
):
    device=model.device
    input_t = tokenizer(text=text, max_length=MAX_LENGTH, truncation=True, return_tensors="pt", padding=True).to(device)
    cos_sim_list=[]
    for frame_path in frame_path_list:
        image=Image.open(frame_path)
        input_f = tokenizer(images=image, return_tensors="pt", padding=True).to(device)
        output_t = model.get_text_features(**input_t).flatten()
        output_f = model.get_image_features(**input_f).flatten()
        cos_sim = F.cosine_similarity(output_t, output_f, dim=0).item()
        cos_sim_list.append(cos_sim)
    clip_score_avg=np.mean(cos_sim_list)
    ans=[]
    if clip_score_avg < CLIP_POINT_LOW:
        ans=[1]*NUM_ASPECT
    elif clip_score_avg >= CLIP_POINT_LOW and clip_score_avg < CLIP_POINT_MID:
        ans=[2]*NUM_ASPECT
    elif clip_score_avg >= CLIP_POINT_MID and clip_score_avg < CLIP_POINT_HIGH:
        ans=[3]*NUM_ASPECT
    else:
        ans=[4]*NUM_ASPECT
    return clip_score_avg, ans


def x_clip_score(
    model, 
    tokenizer, 
    processor, 
    text:str,
    frame_path_list:List[str],
):
    
    def _read_video_frames(frame_paths, max_frames):
        total_frames = len(frame_paths)
        indices = np.linspace(0, total_frames - 1, num=max_frames).astype(int)

        selected_frames = [np.array(Image.open(frame_paths[i])) for i in indices]
        return np.stack(selected_frames)
    
    input_text = tokenizer([text], max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt")
    text_feature = model.get_text_features(**input_text).flatten()

    video=_read_video_frames(frame_path_list,MAX_NUM_FRAMES)
    
    input_video = processor(videos=list(video), return_tensors="pt")
    video_feature = model.get_video_features(**input_video).flatten()
    cos_sim=F.cosine_similarity(text_feature, video_feature, dim=0).item()
    ans=[]
    if cos_sim < X_CLIP_POINT_LOW:
        ans=[1]*NUM_ASPECT
    elif cos_sim >= X_CLIP_POINT_LOW and cos_sim < X_CLIP_POINT_MID:
        ans=[2]*NUM_ASPECT
    elif cos_sim >= X_CLIP_POINT_MID and cos_sim < X_CLIP_POINT_HIGH:
        ans=[3]*NUM_ASPECT
    else:
        ans=[4]*NUM_ASPECT
    return cos_sim, ans

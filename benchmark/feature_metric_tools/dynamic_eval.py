import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage import io, color

ROUND_DIGIT=3
DYN_SAMPLE_STEP=4
NUM_ASPECT=5

SSIM_POINT_HIGH=0.9
SSIM_POINT_MID=0.7
SSIM_POINT_LOW=0.5

MSE_POINT_HIGH=3000
MSE_POINT_MID=1000
MSE_POINT_LOW=100

def dynamic_ssim(frame_path_list):
    ssim_list=[]
    sampled_list = frame_path_list[::DYN_SAMPLE_STEP]
    for f_idx in range(len(sampled_list)-1):
        frame_1=Image.open(sampled_list[f_idx])
        frame_1_gray=color.rgb2gray(frame_1)
        frame_2=Image.open(sampled_list[f_idx+1])
        frame_2_gray=color.rgb2gray(frame_2)

        ssim_value, _ = ssim(frame_1_gray, frame_2_gray, full=True,\
                                 data_range=frame_2_gray.max() - frame_2_gray.min())
        ssim_list.append(ssim_value)
    ssim_avg=np.mean(ssim_list)
    
    ans=[]
    if ssim_avg >= SSIM_POINT_HIGH:
        ans=[1]*NUM_ASPECT
    elif ssim_avg <= SSIM_POINT_HIGH and ssim_avg > SSIM_POINT_MID:
        ans=[2]*NUM_ASPECT
    elif ssim_avg <= SSIM_POINT_MID and ssim_avg > SSIM_POINT_LOW:
        ans=[3]*NUM_ASPECT
    else:
        ans=[4]*NUM_ASPECT
    return ssim_avg, ans


def dynamic_mse(frame_path_list):
    mse_list=[]
    sampled_list = frame_path_list[::DYN_SAMPLE_STEP]
    for f_idx in range(len(sampled_list)-1):        
        imageA = cv2.imread(sampled_list[f_idx])
        imageB = cv2.imread(sampled_list[f_idx+1])
        
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        mse_value = err
        mse_list.append(mse_value)
    mse_avg=np.mean(mse_list)
    ans=[]
    if mse_avg >= MSE_POINT_HIGH:
        ans=[4]*NUM_ASPECT
    elif mse_avg < MSE_POINT_HIGH and mse_avg >= MSE_POINT_MID:
        ans=[3]*NUM_ASPECT
    elif mse_avg < MSE_POINT_MID and mse_avg >= MSE_POINT_LOW:
        ans=[2]*NUM_ASPECT
    else:
        ans=[1]*NUM_ASPECT
        
    return mse_avg, ans


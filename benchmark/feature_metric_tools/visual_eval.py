from brisque import BRISQUE
from pypiqe import piqe
from PIL import Image
import numpy as np
from typing import List

ROUND_DIGIT=3
NUM_ASPECT=5

PIQE_POINT_LOW=15
PIQE_POINT_MID=30
PIQE_POINT_HIGH=50

BRISQUE_POINT_LOW=10
BRISQUE_POINT_MID=30
BRISQUE_POINT_HIGH=50

def piqe_output(frame_path_list:List[str],):
    piqe_list=[]
    for frame_path in frame_path_list:
        frame=np.array(Image.open(frame_path))
        piqe_score, _,_,_ = piqe(frame)
        piqe_list.append(piqe_score)
    piqe_avg=np.mean(piqe_list)
    ans=[]
    if piqe_avg < PIQE_POINT_LOW:
        ans=[4]*NUM_ASPECT
    elif piqe_avg < PIQE_POINT_MID:
        ans=[3]*NUM_ASPECT
    elif piqe_avg < PIQE_POINT_HIGH:
        ans=[2]*NUM_ASPECT
    else:
        ans=[1]*NUM_ASPECT
    return piqe_avg, ans


def brisque_output(frame_path_list:List[str],):
    brisque_list=[]
    for frame_path in frame_path_list:
        frame=Image.open(frame_path)
        brisque_score=BRISQUE().score(frame)
        brisque_list.append(brisque_score)
    brisque_avg=np.mean(brisque_list)
    ans=[]
    if brisque_avg < BRISQUE_POINT_LOW:
        ans=[4]*NUM_ASPECT
    elif brisque_avg < BRISQUE_POINT_MID:
        ans=[3]*NUM_ASPECT
    elif brisque_avg < BRISQUE_POINT_HIGH:
        ans=[2]*NUM_ASPECT
    else:
        ans=[1]*NUM_ASPECT
    return brisque_avg, ans
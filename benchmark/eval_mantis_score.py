import os
import json
import re
import torch
import fire
import logging
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List
from transformers import AutoProcessor
from models.idefics2 import Idefics2ForSequenceClassification
from datasets import load_dataset
from datetime import datetime
from utils_tools import _add_to_res_file,regression_query_template
from utils_conv import conv_templates


CONV_TEMPLATE = conv_templates["idefics_2"]
NUM_ASPECT = 5
ROUND_DIGIT = 4
MAX_NUM_FRAMES = 16
REGRESSION_QUERY_TEMPLATE = regression_query_template()


def _read_video_frames(
    frame_paths:List[str], 
    max_frames:int,
):
    
    total_frames = len(frame_paths)
    indices = np.linspace(0, total_frames - 1, num=max_frames).astype(int)
    selected_frames = [np.array(Image.open(frame_paths[i])) for i in indices]
    return np.stack(selected_frames)


def _model_output(
    model: Idefics2ForSequenceClassification,
    processor: AutoProcessor,
    video_prompt: str, 
    frames_path_list: List[str],
):
    
    video_frames=_read_video_frames(frames_path_list,MAX_NUM_FRAMES)
    frames = [Image.fromarray(x) for x in video_frames]
    eval_prompt = REGRESSION_QUERY_TEMPLATE.format(text_prompt=video_prompt)
    num_image_token = eval_prompt.count("<image>")
    if num_image_token < len(frames):
        eval_prompt += "<image> " * (len(frames) - num_image_token)
    
    if not eval_prompt:
        print("Please provide a prompt")
        return
    if not [frames]:
        frames = None
    
    flatten_images = []
    for x in [frames]:
        if isinstance(x, list):
            flatten_images.extend(x)
        else:
            flatten_images.append(x)
    flatten_images = [Image.open(x) if isinstance(x, str) else x for x in flatten_images]
    inputs = processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    num_aspects = logits.shape[-1]
    
    aspect_scores = []
    for i in range(num_aspects):
        aspect_scores.append(round(logits[0, i].item(),ROUND_DIGIT))
    return aspect_scores


def _cal_spearman_correlation(
    res_file: str="./benchmark/eval_results/eval_videofb_mantisscore.json",
    bench_name: str="video_feedback"
):
    all_res=json.load(open(res_file,"r"))
    all_ref_scores=[eval(item["ref"]) for item in all_res]
    all_ans_scores=[eval(item["ans"]) for item in all_res]
    
    spearman_list=[]
    p_value_list=[]
    try:
        all_ref_scores=np.array(all_ref_scores)
        all_ans_scores=np.array(all_ans_scores)
        for i in range(len(all_ref_scores[0])):
            ref_list=[x[i] for x in all_ref_scores]
            ans_list=[x[i] for x in all_ans_scores]
            rho,p_value=stats.spearmanr(ref_list,ans_list)
            if not np.isnan(rho):
                rho*=100
                spearman_list.append(round(rho,ROUND_DIGIT))
                p_value_list.append(p_value)
            else:
                spearman_list.append(None)
                p_value_list.append(None)    
    except Exception as e:
        logger.info(e)
        spearman_list=[None for _ in range(len(all_ref_scores[0]))]
        p_value_list=[None for _ in range(len(all_ref_scores[0]))]

    dirname=os.path.dirname(res_file)
    with open(f"{res_file}/spearman_corr_{bench_name}.json") as file:
        json.dump({
            "spearman_list":spearman_list,
            "p_value_list":p_value_list,
        },file,indent=4)


def _cal_pairwise_acc(
    res_file: str="./benchmark/eval_results/eval_genaibench_mantisscore.json",
    bench_name: str="genaibench"
):
    None


def main(
    model_repo_name: str="TIGER-Lab/MantisScore",
    data_repo_name: str="TIGER-Lab/MantisScore-Bench",
    frames_dir: str="./data/videofb/test", 
    name_postfixs: List[str]=['video_feedback'], 
    result_file: str="./benchmark/eval_results/eval_videofb_mantisscore.json",
    bench_name: str="video_feedback"
):
    '''
    evalualte MantisScore model on MantisScore-Bench which contains four benchmarks, save results to 'result_file' 
    and calculate spearman correlation coefficient between human-annotated references and model output.
    '''
    
    processor = AutoProcessor.from_pretrained(model_repo_name,torch_dtype=torch.bfloat16)
    model = Idefics2ForSequenceClassification.from_pretrained(model_repo_name,torch_dtype=torch.bfloat16).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("processor and model loaded")
    
    for source in name_postfixs:
        test_data=load_dataset(data_repo_name,name=source, split="test")
        
        curr_frames_dir=f"{frames_dir}/frames_{source}"
        
        for idx, item in tqdm(enumerate(test_data)):
            vid=item["id"]
            frame_path_list=[f"{curr_frames_dir}/{vid}/{img}" for img in item["images"]]
            
            human_text=item["conversations"][0]["value"]
            bot_text=item["conversations"][1]["value"]
            
            video_prompt=human_text.split("text prompt is \"")[1].split("\",\n")[0]

            ref_scores=[int(item) for item in re.findall(r': (\d+)', bot_text)]

            ans_scores=_model_output(model, processor, video_prompt, frame_path_list)

            logger.info(f"{idx} {vid} {ans_scores}")
            curr_compare_dict={
                "id":vid,
                "text":video_prompt,
                "ref":f"{ref_scores}",
                "ans":f"{ans_scores}"
            }
            _add_to_res_file(result_file,curr_compare_dict)
            
    if bench_name in ["video_feedback","eval_crafter"]:      
        _cal_spearman_correlation(result_file,bench_name)
    elif bench_name in ["genaibench","vbench"]:
        _cal_pairwise_acc(result_file,bench_name)
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger= logging.getLogger(__name__)
    date_time=datetime.now().strftime("%m-%d %H:%M:%S")
    log_file=f"./benchmark/logs/eval_on_videofb_{date_time}.log"
    os.makedirs(os.path.dirname(log_file),exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    fire.Fire(main)

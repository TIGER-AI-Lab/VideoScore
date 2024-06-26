import os
import re
import torch
import fire
import logging
from tqdm import tqdm
from typing import List
from datasets import load_dataset
from datetime import datetime
from utils_tools import _add_to_res_file

BENCH_NAMES=["video_feedback","eval_crafter","vbench","genaibench"]


def main(
    data_repo_name: str="TIGER-Lab/VideoScore-Bench",
    metric_name: str="CLIP-Score",
    bench_name: str="video_feedback",
    frames_dir: str="../data/video_feedback/test", 
    name_postfixs: List[str]=['video_feedback'], 
    result_file: str="./eval_results/video_feedback/eval_video_feedback_CLIP-Score.json",
    
):
    print("metric_name",metric_name)
    
    if metric_name in ["PIQE","BRISQUE"]:
        from feature_metric_tools.visual_eval import piqe_output, brisque_output
    elif metric_name in ["CLIP-sim","DINO-sim","SSIM-sim"]:
        from feature_metric_tools.temporal_eval import clip_inter_frame, dino_inter_frame, ssim_inter_frame
    elif metric_name in ["MSE-dyn","SSIM-dyn"]:
        from feature_metric_tools.dynamic_eval import  dynamic_mse, dynamic_ssim
    elif metric_name in ["CLIP-Score","X-CLIP-Score"]:
        from feature_metric_tools.t2v_align_eval import clip_score, x_clip_score
    else:
        raise ValueError("metric name is not supported")
    
    logging.basicConfig(level=logging.INFO)
    logger= logging.getLogger(__name__)
    date_time=datetime.now().strftime("%m-%d %H:%M:%S")
    log_file=f"./logs/{bench_name}/eval_{metric_name}_on_{bench_name}_{date_time}.log"
    os.makedirs(os.path.dirname(log_file),exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,tokenizer,preprocess,processor=None,None,None,None
    
    if metric_name == "CLIP-sim" or "CLIP-Score":
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    if metric_name == "DINO-sim":
        from torchvision.models import vit_b_16 
        import torchvision.transforms as transforms
        model = vit_b_16(pretrained=True)
        model.to(device)
        model.eval()  
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    if metric_name == "X-CLIP-Score":
        from transformers import AutoTokenizer, AutoModel, AutoProcessor
        
        model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
        processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        
    for source in name_postfixs:
        test_data=load_dataset(data_repo_name,name=source, split="test")
        
        curr_frames_dir=f"{frames_dir}/frames_{source}"
        
        for idx, item in tqdm(enumerate(test_data)):
            vid=item["id"]
            frame_path_list=[f"{curr_frames_dir}/{vid}/{img}" for img in item["images"]]
            
            human_text=item["conversations"][0]["value"]
            bot_text=item["conversations"][1]["value"]
            
            video_prompt=human_text.split("text prompt is \"")[1].split("\",\n")[0]
            assert bench_name in BENCH_NAMES, "benchmark name is not supported"
            if bench_name=="video_feedback":
                ref_scores=[int(item) for item in re.findall(r': (\d+)', bot_text)]
            else:
                ref_scores=item["score_list"]
            
            if metric_name=="PIQE":
                raw_float_score, ans_scores = piqe_output(frame_path_list)
            elif metric_name == "BRISQUE":
                raw_float_score, ans_scores = brisque_output(frame_path_list)
            elif metric_name == "CLIP-sim":
                raw_float_score, ans_scores = clip_inter_frame(model, tokenizer, frame_path_list)
            elif metric_name == "DINO-sim":
                raw_float_score, ans_scores = dino_inter_frame(model, preprocess, frame_path_list)
            elif metric_name == "SSIM-sim":
                raw_float_score, ans_scores = ssim_inter_frame(frame_path_list)
            elif metric_name == "MSE-dyn":
                raw_float_score, ans_scores = dynamic_mse(frame_path_list)
            elif metric_name == "SSIM-dyn":
                raw_float_score, ans_scores = dynamic_ssim(frame_path_list)
            elif metric_name == "CLIP-Score":
                raw_float_score, ans_scores = clip_score(model,tokenizer,video_prompt,frame_path_list)
            elif metric_name == "X-CLIP-Score":
                raw_float_score, ans_scores = x_clip_score(model, tokenizer, processor,video_prompt,frame_path_list)
            else:
                raise ValueError("Metric not supported")
            
            logger.info(f"{idx} {vid} {ans_scores}")
            curr_compare_dict={
                "id":vid,
                "text":video_prompt,
                "raw_score":raw_float_score,
                "ref":f"{ref_scores}",
                "ans":f"{ans_scores}"
            }
            _add_to_res_file(result_file,curr_compare_dict)
                

if __name__ == "__main__":
    fire.Fire(main)

import re
import os
import torch
from tqdm import tqdm
from typing import List
from datetime import datetime
import fire
import logging
from datasets import Dataset, load_dataset
from utils_tools import _add_to_res_file, _ans_formatted, merge_images
import multiprocessing

ROUND_DIGIT=3
NUM_ASPECT=5
MAX_TRY_FOR_BAD_MODEL=20
MAX_NUM_FRAMES=24
BENCH_NAMES=["video_feedback","eval_crafter","vbench","genaibench"]
MODEL_NAMES=["llava_next","llava","idefics1","idefics2","kosmos2","openflamingo","cogvlm","fuyu"]

NO_IMG_TOKENS=["idefics2","llava","llava_next"]
GOOD_MODELS=["idefics1","llava","llava_next",]

def _limit_image_tags(input_string, N):
    image_tags = re.findall(r'<image>', input_string)
    if len(image_tags) > N:
        excess_count = len(image_tags) - N
        result_string = re.sub(r'<image>', '', input_string, excess_count)
        return result_string
    else:
        return input_string


def _model_output(model, model_name, img_url_list,input_prompt):
    
    if model_name in NO_IMG_TOKENS:
        input_prompt=_limit_image_tags(input_prompt,0)
    
    if len(img_url_list)>MAX_NUM_FRAMES:
        step = len(img_url_list) / MAX_NUM_FRAMES
        img_url_list = [img_url_list[int(i * step)] for i in range(MAX_NUM_FRAMES)]
    input_prompt=_limit_image_tags(input_prompt,MAX_NUM_FRAMES)
        
    with torch.no_grad():
        inputs=[{
            "type": "image",
            "content": img_url
        } for img_url in img_url_list]+[{
            "type": "text",
            "content": input_prompt
        }]
        output= model(inputs)
    return output


def query_one_video(model, model_name, bench_name, item, logger):  
    aws_s3_prefix="https://video-bench-800.s3.ap-southeast-2.amazonaws.com/frames"
    if bench_name=="eval_crafter":
        aws_s3_prefix=f"https://eval-crafter-frames.s3.ap-southeast-2.amazonaws.com/frames"
    if bench_name=="vbench":
        aws_s3_prefix=f"https://vbench-frames.s3.ap-southeast-2.amazonaws.com/frames"
    if bench_name=="genaibench":
        aws_s3_prefix="https://genaibench-frames.s3.ap-southeast-2.amazonaws.com/frames"
        
    vid=item["id"]
    img_name_list=[f"{vid}/{img_file}" for img_file in item["images"]]
    img_url_list=[f"{aws_s3_prefix}/{vid}/{img_name}" for img_name in item["images"]]
    print(f"len of img_name_list",len(img_name_list))
    human_text=item["conversations"][0]["value"]
    bot_text=item["conversations"][1]["value"]

    video_prompt=human_text.split("text prompt is \"")[1].split("\",\nall")[0]
    ref_scores=[int(item) for item in re.findall(r': (\d+)', bot_text)]
        
    output=_model_output(model,model_name,img_name_list,video_prompt,)
    
    ans_scores=[int(item) for item in re.findall(r': (\d+)', output)]
    ans_scores=_ans_formatted(ans_scores,NUM_ASPECT)
    logger.info(f"{vid} {ans_scores}\n")
    
    curr_compare_dict={
        "id":vid,
        "text":video_prompt,
        "ref":f"{ref_scores}",
        "ans":f"{ans_scores}"
    }
    
    return curr_compare_dict


def main(
    data_repo_name: str="TIGER-Lab/VideoScore-Bench",
    bench_name: str="video_feedback",
    name_postfixs: List[str]=['video_feedback'], 
    result_file: str="./eval_results/video_feedback/eval_video_feedback_llava_next.json",
    model_name: str="llava_next",
):
    if model_name not in MODEL_NAMES:
        raise ValueError("the model is not supported")
    
    logging.basicConfig(level=logging.INFO)
    logger= logging.getLogger(__name__)
    date_time=datetime.now().strftime("%m-%d %H:%M:%S")
    log_file=f"./logs/eval_{model_name}_on_{bench_name}_{date_time}.log"
    os.makedirs(os.path.dirname(log_file),exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    
    aws_s3_prefix="https://video-bench-800.s3.ap-southeast-2.amazonaws.com/frames"
    if bench_name=="eval_crafter":
        aws_s3_prefix=f"https://eval-crafter-frames.s3.ap-southeast-2.amazonaws.com/frames"
    if bench_name=="vbench":
        aws_s3_prefix=f"https://vbench-frames.s3.ap-southeast-2.amazonaws.com/frames"
    if bench_name=="genaibench":
        aws_s3_prefix="https://genaibench-frames.s3.ap-southeast-2.amazonaws.com/frames"

    # MLLM with GOOD output
    if model_name == "idefics1":
        from mllm_tools.idefics1_eval import Idefics1
        model = Idefics1()

    elif model_name == "llava":
        from mllm_tools.llava_eval import Llava
        model = Llava()
    elif model_name == "llava_next":
        from mllm_tools.llava_next_eval import LlavaNext
        model = LlavaNext()
        
    # MLLM with BAD output
    elif model_name == "idefics2": 
        from mllm_tools.idefics2_eval import Idefics2
        model = Idefics2()
    elif model_name == "kosmos2":
        from mllm_tools.kosmos2_eval import Kosmos2
        model=Kosmos2()
    elif model_name == "fuyu":
        from mllm_tools.fuyu_eval import Fuyu
        model = Fuyu()
    elif model_name == "cogvlm":
        from mllm_tools.cogvlm_eval import CogVLM
        model = CogVLM("THUDM/cogvlm-chat-hf")
    elif model_name == "openflamingo":
        from mllm_tools.openflamingo_eval import OpenFlamingo
        model = OpenFlamingo()
    elif model_name == "otterimage":
        from mllm_tools.otterimage_eval import OtterImage
        model = OtterImage()
    else:
        raise ValueError(f"model '{model_name}' is not supported")
    
    
    test_data=[]
    for source in name_postfixs:         
        test_data=load_dataset(data_repo_name,name=source, split="test")
        
        # test_dataset= Dataset.from_list(test_data)
        # ans_dataset=test_dataset.map(lambda x: query_one_video(model, model_name, bench_name, item, logger),num_proc=8)
        # ans_dataset.save_to_disk(result_file)
        
        # test_dataset= Dataset.from_list(test_data)
        # with multiprocessing.Pool(8) as pool:
        #     # results = pool.map(lambda x: query_one_video(model, model_name, bench_name, item, logger), test_dataset)
        #     results = pool.starmap(query_one_video, [(example, source) for example in test_dataset])
        # ans_dataset=Dataset.from_list(results)
        # ans_dataset.save_to_disk(result_file)
        
        for idx,item in tqdm(enumerate(test_data)):
            vid=item["id"]
            img_url_list=[f"{aws_s3_prefix}/{vid}/{img_name}" for img_name in item["images"]]

            human_text=item["conversations"][0]["value"]
            bot_text=item["conversations"][1]["value"]

            video_prompt=human_text.split("text prompt is \"")[1].split("\",\nall")[0]
            
            assert bench_name in BENCH_NAMES, "benchmark name is not supported"
            if bench_name=="video_feedback":
                ref_scores=[int(item) for item in re.findall(r': (\d+)', bot_text)]
            else:
                ref_scores=item["score_list"]
                
            output=_model_output(model, model_name,img_url_list,human_text)
            logger.info(f"{idx} {vid} {output}")
            
            ans_scores=[int(item) for item in re.findall(r': (\d+)', output)]
            ans_scores=_ans_formatted(ans_scores,NUM_ASPECT)
            
            logger.info(f"{idx} {vid} {ans_scores}")
            curr_compare_dict={
                "id":vid,
                "text":video_prompt,
                "ref":f"{ref_scores}",
                "ans":f"{ans_scores}"
            }
            _add_to_res_file(result_file,curr_compare_dict)
            
            if idx>MAX_TRY_FOR_BAD_MODEL and model_name not in GOOD_MODELS:
                return


if __name__=="__main__":
    fire.Fire(main())

import google.generativeai as genai
import fire
from datasets import load_dataset
from typing import List
from PIL import Image
import re
from tqdm import tqdm
from datetime import datetime
import os
import time
import logging
from utils_tools import _add_to_res_file, _ans_formatted


ROUND_DIGIT=4
NUM_ASPECT=5
MAX_TRY=5
MAX_NUM_FRAMES=16
BENCH_NAMES=["video_feedback","eval_crafter","vbench","genaibench"]
MODEL_NAMES=["gemini-1.5-pro-latest","gemini-1.5-flash-latest"]

def eval_gemini(
    data_repo_name: str="TIGER-Lab/VideoScore-Bench",
    bench_name: str="video_feedback",
    frames_dir: str="../data/video_feedback/test", 
    name_postfixs: List[str]=['video_feedback'], 
    result_file: str="./eval_results/video_feedback/eval_video_feedback_gemini-1.5-pro.json",
    base_model: str="gemini-1.5-pro-latest",
):
    
    if base_model not in MODEL_NAMES:
        raise ValueError("gemini base model is not supported")
    
    model = genai.GenerativeModel(base_model, generation_config=generation_config, safety_settings=safety_settings)
    
    logging.basicConfig(level=logging.INFO)
    logger= logging.getLogger(__name__)
    date_time=datetime.now().strftime("%m-%d %H:%M:%S")
    log_file=f"./logs/eval_{base_model}_on_{bench_name}_{date_time}.log"
    os.makedirs(os.path.dirname(log_file),exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    

    test_data=[]
    for source in name_postfixs:
        test_data=load_dataset(data_repo_name,name=source, split="test")
        curr_frames_dir=f"{frames_dir}/frames_{source}"
        for idx, item in tqdm(enumerate(test_data)):
            retries=0
            while retries < MAX_TRY:
                try:
                    vid=item["id"]
                    
                    img_path_list=[f"{curr_frames_dir}/{vid}/{img_file}" for img_file in item["images"]]
                    human_text=item["conversations"][0]["value"]
                    bot_text=item["conversations"][1]["value"]

                    video_prompt=human_text.split("text prompt is \"")[1].split("\",\nall")[0]
                    
                    assert bench_name in BENCH_NAMES, "benchmark name is not supported"
                    if bench_name=="video_feedback":
                        ref_scores=[int(item) for item in re.findall(r': (\d+)', bot_text)]
                    else:
                        ref_scores=item["score_list"]
                    
                    img_list=[]
                    for img_path in img_path_list:
                        img=Image.open(img_path)
                        img_list.append(img)
                    inputs=[human_text]+img_list
                    chat_session = model.start_chat(
                    history=[]
                    )
                    response = chat_session.send_message(inputs)
                    response=response._result.candidates[0].content.parts
                    output=f"{response}".split('[text: "')[1][:-3]
                    logger.info(f"{idx} {vid} {output}")

                    ans_scores=[int(item) for item in re.findall(r'(?:: |:\*\* )(\d+)', output)]
                    ans_scores=_ans_formatted(ans_scores,NUM_ASPECT)
                    logger.info(f"{idx} {vid} {ans_scores}")

                    curr_compare_dict={
                        "id":vid,
                        "text":video_prompt,
                        "ref":f"{ref_scores}",
                        "ans":f"{ans_scores}"
                    }
                    _add_to_res_file(result_file,curr_compare_dict)
                    break
                except Exception as e:
                    logger.info(e)
                    if "OTHER" in f"{e}":
                        curr_compare_dict={
                            "id":vid,
                            "text":video_prompt,
                            "ref":f"{ref_scores}",
                            "ans":f"[-1, -1, -1, -1, -1]"
                        }
                        _add_to_res_file(result_file,curr_compare_dict)
                        break
                    else:
                        logger.info("sleeping for 30 seconds")
                        time.sleep(30)
                        retries += 1
                        logger.error(f"retrying for {retries} time")

    

if __name__=="__main__":
    
    api_key=os.getenv("GEMINI_API_KEY",None)
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 1000,
    }
    safety_settings = [
        {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
        },
        {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
        },
        {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
        },
        {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
        },
    ]
    

    fire.Fire(eval_gemini)
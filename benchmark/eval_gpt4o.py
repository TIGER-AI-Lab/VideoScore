import base64
import os
import time
import fire
import json
import re
import logging
from typing import List
from string import Template
from tqdm import tqdm
from datetime import datetime
from datasets import Dataset, load_dataset
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from utils_tools import _add_to_res_file, _ans_formatted, label_query_template
from utils_gpt4o import GPT4o


ROUND_DIGIT = 3
NUM_ASPECT = 5
MAX_TRY = 5
MAX_NUM_FRAMES = 20
BENCH_NAMES=["video_feedback","eval_crafter","vbench","genaibench"]
QUERY_TEMPLATE = label_query_template()


def _shorten_image_list(input_list,max_num_frame):
    while len(input_list) > max_num_frame:
        step = len(input_list) // max_num_frame
        input_list = [item for index, item in enumerate(input_list) if index % step != 0]
    return input_list

def _azure_gpt_4o_output(img_name_list,video_prompt,aws_s3_prefix):

    gpt_class=GPT4o()
    template=Template(QUERY_TEMPLATE)
    input_text_prompt=template.substitute(source=video_prompt,num_aspect=NUM_ASPECT)
    img_url_list=[f"{aws_s3_prefix}/{img_name}" for img_name in img_name_list]
    
    if len(img_url_list)>MAX_NUM_FRAMES:
        step = len(img_url_list) / MAX_NUM_FRAMES
        img_url_list = [img_url_list[int(i * step)] for i in range(MAX_NUM_FRAMES)]

    prompt_content=gpt_class.prepare_prompt(img_url_list, input_text_prompt)
    messages=[
        {
            "role":"user",
            "content":prompt_content
        }
    ]

    completions=client.chat.completions.create(
        model=deployment,
        messages=messages)
    
    res = completions.choices[0].message.content
    return res


def query_one_video(logger,item,aws_s3_prefix):
    vid=item["id"]
    img_name_list=[f"{vid}/{img_file}" for img_file in item["images"]]
    print(f"len of img_name_list",len(img_name_list))
    human_text=item["conversations"][0]["value"]
    bot_text=item["conversations"][1]["value"]

    video_prompt=human_text.split("text prompt is \"")[1].split("\",\nall")[0]
    ref_scores=[int(item) for item in re.findall(r': (\d+)', bot_text)]
        
    output=_azure_gpt_4o_output(img_name_list,video_prompt,aws_s3_prefix)
    
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


def eval_gpt4(
    data_repo_name: str="TIGER-Lab/VideoFeedback-Bench",
    bench_name: str="video_feedback",
    name_postfixs: List[str]=['video_feedback'], 
    result_file: str="./eval_results/video_feedback/eval_video_feedback_gpt4o.json",
):
    logging.basicConfig(level=logging.INFO)
    logger= logging.getLogger(__name__)
    date_time=datetime.now().strftime("%m-%d %H:%M:%S")
    log_file=f"./logs/eval_gpt4o_on_{bench_name}_{date_time}.log"
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
    
    for source in name_postfixs:
        test_data=load_dataset(data_repo_name,name=source, split="test")
        
        # test_dataset = Dataset.from_list(test_data)  
        # ans_dataset=test_dataset.map(query_one_video,num_proc=8)

        for idx,item in tqdm(enumerate(test_data)):
            retries = 0
            while retries < MAX_TRY:
                try:
                    vid=item["id"]
                    img_name_list=[f"{vid}/{img_file}" for img_file in item["images"]]
                    print(f"len of img_name_list",len(img_name_list))
                    human_text=item["conversations"][0]["value"]
                    bot_text=item["conversations"][1]["value"]

                    video_prompt=human_text.split("text prompt is \"")[1].split("\",\nall")[0]
                    assert bench_name in BENCH_NAMES, "benchmark name is not supported"
                    if bench_name=="video_feedback":
                        ref_scores=[int(item) for item in re.findall(r': (\d+)', bot_text)]
                    else:
                        ref_scores=item["score_list"]
                                            
                    output=_azure_gpt_4o_output(img_name_list,video_prompt,aws_s3_prefix)
                    
                    ans_scores=[int(item) for item in re.findall(r': (\d+)', output)]
                    ans_scores=_ans_formatted(ans_scores,NUM_ASPECT)
                    logger.info(f"{vid} {ans_scores}\n")
                    
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
                    logger.info("sleeping for 30 seconds")
                    time.sleep(30)
                    retries += 1
                    logger.error(f"retrying for {retries} time")

            if retries >= MAX_TRY:
                logger.info("quota may run out")
                logger.info(f"total num of evaled videos {len(json.load(open(result_file)))}")
                break
        if retries >= MAX_TRY:
            break
          

if __name__=="__main__":
    ### use azure endpoint
    endpoint=os.getenv("AZURE_END_POINT",None)
    deployment=os.getenv("CHAT_COMPLETIONS_DEPLOYMENT_NAME",None)
    api_key=os.getenv("AZURE_OPENAI_API_KEY",None)
    api_version=os.getenv("AZURE_OPENAI_API_VERSION",None)
    
    token_provider=get_bearer_token_provider(DefaultAzureCredential(),"https://cognitiveservices.azure.com/.default")
    client=AzureOpenAI(azure_endpoint=endpoint,api_key=api_key,api_version=api_version)

    gpt_model_name="gpt_4o_zeroshot"    

    set_name="refined_40k"
    name_test="bench"    
    source_list=["inc","lab","real","fastsvd",'sora',"bad_img","bad_mi2v","bad_insf","bad_insf_static","bad_null_prompt","bad_prompt",'bad_worsen_phy']

    
    fire.Fire(eval_gpt4)


    
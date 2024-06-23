import json
import re
import os
import time
import torch
from PIL import Image
from tqdm import tqdm
from typing import List
from datetime import datetime
import logging
from datasets import Dataset
from utils_tools import _add_to_res_file, _ans_formatted, merge_images
import multiprocessing


def _limit_image_tags(input_string, N):
    image_tags = re.findall(r'<image>', input_string)
    if len(image_tags) > N:
        excess_count = len(image_tags) - N
        result_string = re.sub(r'<image>', '', input_string, excess_count)
        return result_string
    else:
        return input_string


def _model_output(img_url_list,input_prompt):
    if model_name in no_need_image_tokens:
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
        # inputs={k: v.to(model.device) for k, v in inputs.items()}
        output= model(inputs,gpu_idx=int(os.environ["CUDA_VISIBLE_DEVICES"]))
        print(os.environ["CUDA_VISIBLE_DEVICES"])
    logger.info(f"{output}")
    return output


def query_one_video(item,source):
    aws_s3_prefix="https://video-bench-800.s3.ap-southeast-2.amazonaws.com/frames"
    if ECTV_BENCH:
        aws_s3_prefix=f"https://eval-crafter-frames.s3.ap-southeast-2.amazonaws.com/frames_{source}"
    if VBENCH:
        aws_s3_prefix=f"https://vbench-frames.s3.ap-southeast-2.amazonaws.com/frames_{source}"
    if GENAI_BENCH:
        aws_s3_prefix="https://genaibench-frames.s3.ap-southeast-2.amazonaws.com/frames"
            
    vid=item["id"]
    img_name_list=[f"{vid}/{img_file}" for img_file in item["images"]]
    img_url_list=[f"{aws_s3_prefix}/{vid}/{img_name}" for img_name in item["images"]]
    print(f"len of img_name_list",len(img_name_list))
    human_text=item["conversations"][0]["value"]
    bot_text=item["conversations"][1]["value"]

    video_prompt=human_text.split("text prompt is \"")[1].split("\",\nall")[0]
    ref_scores=[int(item) for item in re.findall(r': (\d+)', bot_text)]
        
    output=_model_output(img_name_list,video_prompt,)
    
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


def eval_other_mllm(set_name,name_test,source_list,model_name):
    data_rt_dir="/data/xuan/video_eval"
    
    eval_res_dir=f"{data_rt_dir}/eval_results/{set_name}"
    os.makedirs(eval_res_dir,exist_ok=True)
    eval_res_file=f"{eval_res_dir}/eval_{name_test}_{model_name}.json"
    # acc_list_file=f"{eval_res_dir}/acc_list.json"
    # all_acc_list=[]
    # if os.path.exists(acc_list_file):
    #     all_acc_list=json.load(open(acc_list_file,"r"))
    # else:
    #     with open(acc_list_file,"w") as f:
    #         json.dump([],f)

    all_ref_scores=[]
    all_ans_scores=[]
    res_compare_list=[]
    for source in source_list:         
        test_data=json.load(open(f"{data_rt_dir}/data/{set_name}/test_{name_test}/data_{source}.json","r"))
        frames_dir=f"{data_rt_dir}/data/{set_name}/test_{name_test}/frames_{source}"   
        aws_s3_prefix="https://video-bench-800.s3.ap-southeast-2.amazonaws.com/frames"
        if ECTV_BENCH:
            aws_s3_prefix=f"https://eval-crafter-frames.s3.ap-southeast-2.amazonaws.com/frames_{source}"
            eval_res_file=f"{eval_res_dir}/eval_{name_test}_{model_name}_{source}.json"
        if VBENCH:
            aws_s3_prefix=f"https://vbench-frames.s3.ap-southeast-2.amazonaws.com/frames_{source}"
            eval_res_file=f"{eval_res_dir}/eval_{name_test}_{model_name}_{source}.json"
        if GENAI_BENCH:
            aws_s3_prefix="https://genaibench-frames.s3.ap-southeast-2.amazonaws.com/frames"
        
    
        # test_dataset= Dataset.from_list(test_data)
        # ans_dataset=test_dataset.map(lambda x: query_one_video(x, source=source),num_proc=8)
        # ans_dataset.save_to_disk(eval_res_file)
        
        
        # test_dataset= Dataset.from_list(test_data)
        # with multiprocessing.Pool(8) as pool:
        #     # results = pool.map(lambda x: query_one_video(x, source=source), test_dataset)
        #     results = pool.starmap(query_one_video, [(example, source) for example in test_dataset])
        # ans_dataset=Dataset.from_list(results)
        # ans_dataset.save_to_disk(eval_res_file)
        
        
        
        for idx,item in tqdm(enumerate(test_data)):
            vid=item["id"]
            img_url_list=[f"{aws_s3_prefix}/{vid}/{img_name}" for img_name in item["images"]]

            human_text=item["conversations"][0]["value"]
            bot_text=item["conversations"][1]["value"]

            video_prompt=human_text.split("text prompt is \"")[1].split("\",\nall")[0]
            
            if ECTV_BENCH or GENAI_BENCH or VBENCH:
                ref_scores=item["score_list"]
            else:
                ref_scores=[int(item) for item in re.findall(r': (\d+)', bot_text)]
            
            output=_model_output(img_url_list,human_text)
            ans_scores=[int(item) for item in re.findall(r': (\d+)', output)]
            ans_scores=_ans_formatted(ans_scores,NUM_ASPECT)
            
            logger.info(f"{idx} {vid} {ans_scores}")
            curr_compare_dict={
                "id":vid,
                "text":video_prompt,
                "ref":f"{ref_scores}",
                "ans":f"{ans_scores}"
            }
            _add_to_res_file(eval_res_file,curr_compare_dict)
            res_compare_list.append(curr_compare_dict)
            all_ref_scores.append(ref_scores)
            all_ans_scores.append(ans_scores)
            
            if idx>MAX_SAMPLE and model_name not in good_models:
                return None

    # acc_list=[]
    # match_list=[]
    # for ref,ans in zip(all_ref_scores,all_ans_scores):
    #     match_list.append([1 if ref[i]==ans[i] else 0 for i in range(len(ref))])
    # acc_list=[round(sum(x)/len(x),ROUND_DIGIT) for x in zip(*match_list)]

    # all_acc_list.append({f"test_{name_test}_{model_name}":f"{acc_list}"})
    # with open(acc_list_file,"w") as f:
    #     json.dump(all_acc_list,f,indent=4)
    # return acc_list


if __name__=="__main__":

    ROUND_DIGIT=4
    NUM_ASPECT=5
    MAX_SAMPLE=20

    MAX_NUM_FRAMES=24
    
    ECTV_BENCH=0
    GENAI_BENCH=0
    VBENCH=0
    
    good_models=["llava_next","llava"]
    
    no_need_image_tokens=["idefics2","llava","llava_next"]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    
    
    # from mllm_tools.idefics1_eval import Idefics1
    # model_name="idefics1"
    # model = Idefics1()
    
    
    # from mllm_tools.llava_eval import Llava
    # model_name="llava"
    # model = Llava()
    
    
    from mllm_tools.llava_next_eval import LlavaNext
    model_name="llava_next"
    model = LlavaNext()    
    
    
    # set_name="refined_40k"
    # name_test="bench"    
    # source_list=["inc","lab","real","fastsvd",'sora',"bad_img","bad_mi2v","bad_insf","bad_insf_static","bad_null_prompt","bad_prompt",'bad_worsen_phy']
    
    
    set_name="eval_crafter"
    name_test="ectv_500"
    # source_list=['pika']
    # source_list=['gen2']
    # source_list=['floor33']
    # source_list=['modelscope']
    source_list=['zeroscope']
    ECTV_BENCH=1
    
    # python eval_other_mllm.py
    
    
    # set_name="genaibench"
    # name_test="genaibench"
    # source_list=["genaibench"]
    # GENAI_BENCH=1
    

    
    # set_name="vbench"
    # name_test="vbench"
    # source_list=["technical_quality","subject_consistency","dynamics_degree","motion_smoothness","overall_consistency"]
    # source_list=["technical_quality"]
    # source_list=["subject_consistency"]
    # source_list=["dynamics_degree"]
    # source_list=["motion_smoothness"]
    # source_list=["overall_consistency"]
    # VBENCH=1
    
    
    logging.basicConfig(level=logging.INFO)
    logger= logging.getLogger(__name__)
    date_time=datetime.now().strftime("%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(f'./logs/{set_name}/mllm/eval_{model_name}_{date_time}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    eval_other_mllm(set_name,name_test,source_list,model_name)

    
    

    
    
    
    # from mllm_tools.blip_flant5_eval import BLIP_FLANT5
    # model_name="blip2-flan-t5-xxl"
    # model = BLIP_FLANT5("Salesforce/blip2-flan-t5-xxl")
    
    # ## conda activate cogvlm
    # from mllm_tools.cogvlm_eval import CogVLM
    # model_name="cogvlm-chat-hf"
    # model = CogVLM("THUDM/cogvlm-chat-hf")
    
    # from mllm_tools.emu2_eval import Emu2
    # model_name="emu2"
    # model = Emu2()
    
    # from mllm_tools.fuyu_eval import Fuyu
    # model_name="fuyu"
    # model = Fuyu()
    

    
    # from mllm_tools.idefics2_eval import Idefics2
    # model_name="idefics2"
    # model = Idefics2()
    
    # from mllm_tools.instructblip_eval import INSTRUCTBLIP_FLANT5
    # model_name="instructblip-flan-t5-xxl"
    # model = INSTRUCTBLIP_FLANT5("Salesforce/instructblip-flan-t5-xxl")
        
    # from mllm_tools.kosmos2_eval import Kosmos2
    # model_name="kosmos2"
    # model = Kosmos2()
    
        # from mllm_tools.mfuyu_eval import MFuyu
    # model_name="mfuyu"
    # model = MFuyu()
    
    # from mllm_tools.mllava_eval import MLlava
    # model_name="mllava"
    # model = MLlava()
    
    # from mllm_tools.otterhd_eval import OtterHD
    # model_name="otterhd"
    # model = OtterHD()
    
    # from mllm_tools.otterimage_eval import OtterImage
    # model_name="otterimage"
    # model = OtterImage()
    
    # from mllm_tools.ottervideo_eval import OtterVideo
    # model_name="ottervideo"
    # model = OtterVideo()
    
    # from mllm_tools.qwenVL_eval import QwenVL
    # model_name="qwenvl"
    # model = QwenVL()
    
    # from mllm_tools.videollava_eval import VideoLlava
    # model_name="videollava"
    # model = VideoLlava()
    
    # from mllm_tools.vila_eval import VILA
    # model_name="vila"
    # model = VILA()
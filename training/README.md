# Data
## prepare both annotated videos and real videos
- Paste the scripts in current directory to repo of Mantis: ```~/Mantis/data/video_eval/``` and 
change current work directory to this directory.

- First run the following scripts to download the video frames (both annotated videos and real videos)and unzip them.
```bash
bash download_anno.sh
bash download_real.sh
```

- Then run the following two scripts  to curate training data in regression scoring format and in generation scoring format, respectively.
```bash
python prepare_regression.py
```
```bash
python prepare_conv.py
```


## only prepare annotated videos
(1) Paste the scripts in current directory to repo of Mantis: ```~/Mantis/data/video_eval/``` and 
change current work directory to this directory.

(2) First run the following script to download the video frames (only annotated videos) and unzip them.
```bash
bash download_anno.sh
```

(3) Then make mofications in ```prepare_regression.py``` and ```prepare_conv.py``` accordingly to only include data of annotated videos and exclude data of real videos. See details in these two scripts.

(4) Finally run the following two scripts  to curate training data in regression scoring format and in generation scoring format, respectively.
```bash
python prepare_regression.py
```
```bash
python prepare_conv.py
```

# Training
## Regression scoring version of VideoScore
(1) change working directory to ```~/Mantis/mantis/train/```.

(2) modify and confirm data configuration in ```~/Mantis/mantis/train/data_configs/train_video_eval.yaml```
for training regression scoring format, it should be like: 
```yaml
data:  
  - 
    name: "video_eval"
    type: json
    path: "../../data/video_eval/train_regression.json"
    format: classification
    split: train
```

(3) set arguments in training script ```~/Mantis/mantis/train/scripts/train_idefics2_video_eval.sh```: 
```bash
hf_hub_user_name="xxxx" # set this will push the model to your hub after training
max_seq_len=4096
lora_enabled=false
qlora_enabled=false
OUTPUT_DIR="../../checkpoints"
global_batch_size=64
problem_type="regression"
num_labels=5
RUN_NAME="xxxx" # model name will be 'xxxx_${max_seq_len}_${problem_type}'
```

(4) start training
```bash
bash scripts/train_idefics2_video_eval.sh
```

## Generation scoring version of VideoScore
(1) change working directory to ```~/Mantis/mantis/train/```.

(2) for training generation scoring format, it should be 
```yaml
data:  
  - 
    name: "video_eval"
    type: json
    path: "../../data/video_eval/train_conv.json"
    format: chat
    split: train
```
(3) set arguments in training script ```~/Mantis/mantis/train/scripts/train_idefics2_video_eval.sh```, 
```bash
hf_hub_user_name="xxxx" # set this will push the model to your hub after training
max_seq_len=4096
lora_enabled=false
qlora_enabled=false
OUTPUT_DIR="../../checkpoints"
global_batch_size=64
problem_type="generation"
num_labels=5
RUN_NAME="xxxx" # model name will be 'xxxx_${max_seq_len}_${problem_type}'
```

(4) start training
```bash
bash scripts/train_idefics2_video_eval.sh
```

# Ablation Study
In our paper, we do ablation studies in three dimensions:
- data sources: annotated and real videos, or annotated data only
- regression-style or generation-style
- base model: Mantis-Idefics2-8B, Idefics2-8B or VideoLLaVA

The first two are discussed above, here we talk about ablation of base model.

We use Mantis-Idefics2-8B as base model by default, 

(1) for Idefics2-8B, modify "model_name_or_path" in ```~/Mantis/mantis/train/scripts/train_idefics2_video_eval.sh```:
```bash
# model_name_or_path="TIGER-Lab/Mantis-8B-Idefics2"
model_name_or_path="HuggingFaceM4/idefics2-8b"
```

then start training
```bash
bash scripts/train_idefics2_video_eval.sh
```


(2) For VideoLLaVA, first modify and confirm data configurations in ```~/Mantis/mantis/train/data_configs/train_video_eval_videochat.yaml```, it should be
```yaml
data:  
  - 
    name: "video_eval"
    type: json
    path: "../../data/video_eval/train_conv.json"
    format: chat_video
    split: train
    max_num_frames: 8
    video_dir: "../../data/video_eval/images"
```

Then confirm the arguments in the training script ```~/Mantis/mantis/train/scripts/train_video_llava.sh```
```bash
hf_hub_user_name="xxxx" # set this will push the model to your hub after training
max_seq_len=2048
lora_enabled=false
qlora_enabled=false
OUTPUT_DIR="../../checkpoints"
global_batch_size=128
RUN_NAME="xxxx" # model name will be 'xxxx_2048'
```

then start training
```bash
bash scripts/train_video_llava.sh
```

# Acknowledgement
Thanks [Mantis](https://github.com/TIGER-AI-Lab/Mantis/tree/main/mantis/train) for the codebase of training VideoScore. 

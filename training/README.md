** README.md is being updated **

## Data
Paste the three scripts in current directory to repo of Mantis: ~/Mantis/data/video_eval/
first run
```bash
bash download.sh
```
to download the video frames and unzip them.

Then run
```bash
python prepare_regression.py
```
and
```bash
python prepare_conv.py
```
respectively to curate training data in regression scoring format and in generation scoring format for VideoScore.

## Training


## Acknowledgement
Code for training MantisScore is from https://github.com/TIGER-AI-Lab/Mantis/tree/main/mantis/train

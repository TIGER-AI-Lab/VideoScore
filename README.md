# MantisScore
official repo for "MantisScore: A Reliable Fine-grained Metric for Video Generation"

<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/TIGER-AI-Lab/MantisScore">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="https://tiger-ai-lab.github.io/MantisScore/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-blue?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/VideoFeedback">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/spaces/TIGER-Lab/Mantis">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Demo-red?style=flat"></a> 
<a target="_blank" href="https://huggingface.co/TIGER-Lab/MantisScore">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-Tweet-blue?style=flat&logo=twitter"></a>
<br>

## Introduction


## Installation

- for inference
```bash
pip install -e . 
```
- for evaluation
```bash
pip install -e .[eval] 
```
- for training
```bash
git clone https://github.com/TIGER-AI-Lab/Mantis
cd Mantis
pip install -e .[train,eval]
pip install flash-attn --no-build-isolation
# then training scripts are in Mantis/train/scripts
```

## Dataset
- [🤗 VideoFeedback](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback) VideoFeedback contains a total of 37.6K text-to-video pairs from 11 popular video generative models, with some real-world videos as data augmentation. The videos are annotated by raters for five evaluation dimensions: Visual Quality, Temporal Consistency, Dynamic Degree, Text-to-Video Alignment and Factual Consistency, in 1-4 scoring scale. 

- [🤗 VideoFeedback-Bench](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback-Bench) 
We derive four test sets from 
VideoFeedback, 
[EvalCrafter](https://github.com/evalcrafter/EvalCrafter), 
[GenAI-Bench](https://huggingface.co/datasets/TIGER-Lab/GenAI-Bench) and 
[VBench](https://github.com/Vchitect/VBench) respectively to curate VideoFeedback-Bench. 
VideoFeedback-Bench is composed of about 7,000 videos, covering both Likert-scale annotation and human preference data.  

## Model
- [🤗 MantisScore](https://huggingface.co/TIGER-Lab/MantisScore) MantisScore is a video quality evaluation model, taking [Mantis-8B-Idefics2](https://huggingface.co/TIGER-Lab/Mantis-8B-Idefics2) as base-model and trained on [VideoFeedback](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback). 

- [🤗 MantisScore-anno-only](https://huggingface.co/TIGER-Lab/MantisScore-anno-only) MantisScore-anno-only is a variant of MantisScore, trained on VideoFeedback with the real videos excluded.


## Inference examples
```bash
cd examples
python run_mantiscore.py
```

## Evaluation
For details, please check [benchmark/README.md](.benchmark/README.md)

## Training
For details, please check [training/README.md](.training/README.md)

## Acknowledgement
- Thanks [Mantis](https://github.com/TIGER-AI-Lab/Mantis/tree/main) for the training codebase of MantisScore and its variants and also for the MLLM plug-and-play tools in evaluation stage! 

- Thanks [VIEScore](https://github.com/TIGER-AI-Lab/VIEScore/tree/main) for some codes of prompting MLLM in evaluation! 

## Citation
```bibtex
```

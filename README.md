# MantisScore
official repo for "MantisScore: A Reliable Fine-grained Metric for Video Generation"


** The code is being updated **


<a target="_blank" href="https://arxiv.org/abs/2406.15252">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/TIGER-AI-Lab/MantisScore">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="https://tiger-ai-lab.github.io/MantisScore/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-blue?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/VideoFeedback">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/spaces/TIGER-Lab/Mantis">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Demo-red?style=flat"></a> 
<a target="_blank" href="https://huggingface.co/collections/TIGER-Lab/mantisscore-6678c9451192e834e91cc0bf">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-Tweet-blue?style=flat&logo=twitter"></a>
<br>

## Introduction

🚀The recent years have witnessed great advances in video generation. However, the development of automatic video metrics is lagging significantly behind. None of the existing metric is able to provide reliable scores over generated videos. 
🤔The main barrier is the lack of large-scale human-annotated dataset.

- 🛢️**VideoFeedback Dataset**. In this paper, we release VideoFeedback, the first large-scale dataset containing human-provided multiaspect score over 37.6K synthesized videos from 11 existing video generative models.

- 🏅**MantisScore**. We train MantisScore (initialized from Mantis) based on VideoFeedback to enable automatic video quality assessment. Experiments show that the Spearman correlation between MantisScore and humans can reach 77.1 on VideoFeedback-test, beating the prior best metrics by about 50 points. Further result on other held-out EvalCrafter, GenAI-Bench, and VBench show that MantisScore has consistently much higher correlation with human judges than other metrics.

- 🫡**Human Feedback for Video generative models**. Due to these results, we believe MantisScore can serve as a great proxy for human raters to (1) rate different video models to track progress (2) simulate fine-grained human feedback in Reinforcement Learning with Human Feedback (RLHF) to improve current video generation models.

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
For details, please check [benchmark/README.md](benchmark/README.md)

## Training
For details, please check [training/README.md](training/README.md)

## Acknowledgement
- Thanks [Mantis](https://github.com/TIGER-AI-Lab/Mantis/tree/main) for the training codebase of MantisScore and its variants and also for the MLLM plug-and-play tools in evaluation stage! 

- Thanks [VIEScore](https://github.com/TIGER-AI-Lab/VIEScore/tree/main) for some codes of prompting MLLM in evaluation! 

## Citation
```bibtex
@article{he2024mantisscore,
  title = {MantisScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation},
  author = {He, Xuan and Jiang, Dongfu and Zhang, Ge and Ku, Max and Soni, Achint and Siu, Sherman and Chen, Haonan and Chandra, Abhranil and Jiang, Ziyan and Arulraj, Aaran and Wang, Kai and Do, Quy Duc and Ni, Yuansheng and Lyu, Bohan and Narsupalli, Yaswanth and Fan, Rongqi and Lyu, Zhiheng and Lin, Yuchen and Chen, Wenhu},
  journal = {ArXiv},
  year = {2024},
  volume={abs/2406.15252},
  url = {https://arxiv.org/abs/2406.15252},
}

```

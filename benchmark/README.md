## Overview
We derive four test sets from 
[VideoFeedback](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback), 
[EvalCrafter](https://github.com/evalcrafter/EvalCrafter), 
[GenAI-Bench](https://huggingface.co/datasets/TIGER-Lab/GenAI-Bench) and 
[VBench](https://github.com/Vchitect/VBench), respectively. 

- For VideoFeedback-test, we take Spearman correlation coefficient (SPCC) between model output and human-annotated score on the five evaluation dimensions as performance indicator.

- For EvalCrafter, we choose three evaluation dimensions from five that match EvalCrafter Benchmark (Visual Quality, Temporal Consistency and Text-to-Video Alignment) and calcualte the SPCC between model output and human score.

- Since GenAI-Bench is composed of human preference data that judges which of the two provided videos is
generally better, we use the average score of five dimensions to predict the preference between two given videos. We take pairwise accuracy as performance indicator.

- VBench contains preference data among four videos, similarly we use the average score of five dimensions to predict the preference and calculate the pairwise accuracy of prediction.

VideoScore-Bench is composed of about 7,000 videos, covering both Likert-scale annotation and human preference data. 

All the data and video frames we used can be found in [🤗VideoScore-Bench](https://huggingface.co/datasets/TIGER-Lab/VideoScore-Bench)

## Prepare data of video frames
To download the video frames for test, please first set benchmark name in data/download_data.sh: 
```bash
bench_name=video_feedback
# bench_name=genaibench
# bench_name=eval_crafter
# bench_name=vbench
```
then run: 
```
cd data && bash download_data.sh
```

## Evaluation
```
cd benchmark
```
To evaluate either VideoScore or GPT4o or other MLLM prompting methods or some featured-based metrics,
please choose the benchmark you want to evaluate on in the corresponding shell scripts firstly:  
```bash
bench_name="video_feedback"
# bench_name="eval_crafter"
# bench_name="genaibench"
# bench_name="vbench"
```

- To evaluate VideoScore model on certain benchmark,
```bash
bash eval_videoscore.sh
```

- To evaluate GPT-4o on certain benchmark,
```bash
bash eval_gpt4o.sh
```

- To get results from Gemini-1.5-Pro or Gemini-1.5-Flash, you also need to choose the vairant model of Gemini-1.5
```bash
base_model="gemini-1.5-flash-latest"
# base_model="gemini-1.5-pro-latest"
```
then run 
```bash
bash eval_gemini-1.5.sh
```


- To get results from other open-source MLLMs, you need to specify the MLLM to be used in script:
```bash
model_name="idefics1"
# model_name="llava"
# model_name="llava_next"
# model_name="idefics2"
# model_name="cogvlm"
# model_name="fuyu"
# model_name="kosmos2"
# model_name="openflamingo"
# model_name="otterimage"
```  
then run 
```bash
bash eval_other_mllm.sh
```

- To get results from the feature-based metrics like CLIP-Score, SSIM, etc, 
you need to specify the metric name in script:
```bash
metric_name="PIQE"
# metric_name="BRISQUE"
# metric_name="CLIP-sim"
# metric_name="DINO-sim"
# metric_name="SSIM-sim"
# metric_name="MSE-dyn"
# metric_name="SSIM-dyn"
# metric_name="CLIP-Score"
# metric_name="X-CLIP-Score"
```
then run 
```bash
bash eval_feature_metric.sh
```

## Get Spearman correlation coefficient or Pairwise Accuracy as performance indicator

After obtaining model output for each video in test set, you can run the following scripts to get SPCC or pairwise accuracy: 
- For VideoFeedback-test or EvalCrafter,

first specify the benchmark name in script:
```bash
bench_name="video_feedback"
# bench_name="eval_crafter"
```
then run
```
bash get_spearman_corr.sh
```

- For results of GenAI-Bench, 
```
bash get_genaibench_pairwise_acc.sh
```

- For results of VBench, 
```
bash get_vbench_pairwise_acc.sh
```

## Check results
For example, the output of MnantisScore on GenAI-Bench set is saved to the following json files, respectively.
```
./eval_results/genaibench/eval_genaibench_mantisscore.json
```

For the Spearman correlation coefficient or the pairwise accuracy, it's saved to 
```
./eval_results/video_feedback/spearman_corr_video_feedback.json
```
```
./eval_results/eval_crafter/spearman_corr_eval_crafter.json
```
```
./eval_results/genaibench/genaibench_pairwise_acc.txt
```
```
./eval_results/vbench/vbench_pairwise_acc.txt
```

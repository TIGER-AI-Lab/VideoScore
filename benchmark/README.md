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
- Evaluate MantisScore on VideoFeedback-test: 
Suppose your current directory is "~/MantisScore"
```
cd benchmark
```

To evaluate model on certain benchmark

//

To be filled

//


After obtaining model output for each video in test set, you can run the following scripts to get SPCC or pairwise accuracy: 
```
bash get_spearman_corr.sh
```
```
bash get_genaibench_pairwise_acc.sh
```
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

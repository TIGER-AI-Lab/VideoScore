
metric_name="PIQE"
# metric_name="BRISQUE"
# metric_name="CLIP-sim"
# metric_name="DINO-sim"
# metric_name="SSIM-sim"
# metric_name="MSE-dyn"
# metric_name="SSIM-dyn"
# metric_name="CLIP-Score"
# metric_name="X-CLIP-Score"

if [ "$metric_name" = "PIQE" ]; then
    if conda info --envs | grep -q "piqe-new"; then
        echo "env piqe exists"
    else
        echo "env piqe not exist, creating..."
        conda create -n piqe-new python=3.10
    fi
    conda activate piqe
    pip install piqe
fi

# # ===== eval MantisScore on VideoFeedback-test =====
# mkdir -p ./eval_results/video_feedback

# model_repo_name="TIGER-Lab/MantisScore"
# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# bench_name="video_feedback"
# frames_dir="../data/${bench_name}/test"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_${metric_name}.json"


# python eval_feature_metric.py --model_repo_name $model_repo_name \
#     --data_repo_name $data_repo_name \
#     --metric_name $metric_name \
#     --bench_name $bench_name \
#     --frames_dir $frames_dir  \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file



# # ===== eval MantisScore on EvalCrafter =====
# mkdir -p ./eval_results/eval_crafter

# # we use this variant "TIGER-Lab/MantisScore-anno-only" except for VideoFeedback-test
# model_repo_name="TIGER-Lab/MantisScore-anno-only"
# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# bench_name="eval_crafter"
# frames_dir="../data/${bench_name}/test"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_${metric_name}.json"

# python eval_feature_metric.py --model_repo_name $model_repo_name \
#     --data_repo_name $data_repo_name \
#     --metric_name $metric_name \
#     --bench_name $bench_name \
#     --frames_dir $frames_dir  \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file




# # ===== eval MantisScore on GenAI-Bench =====
# mkdir -p ./eval_results/genaibench

# # we use this variant "TIGER-Lab/MantisScore-anno-only" except for VideoFeedback-test
# model_repo_name="TIGER-Lab/MantisScore-anno-only"
# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# bench_name="genaibench"
# frames_dir="../data/${bench_name}/test"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_${metric_name}.json"


# python eval_feature_metric.py --model_repo_name $model_repo_name \
#     --data_repo_name $data_repo_name \
#     --metric_name $metric_name \
#     --bench_name $bench_name \
#     --frames_dir $frames_dir  \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file




# ===== eval MantisScore on VBench =====
mkdir -p ./benchmark/eval_results/vbench

# we use this variant "TIGER-Lab/MantisScore-anno-only" except for VideoFeedback-test
model_repo_name="TIGER-Lab/MantisScore-anno-only"
data_repo_name="TIGER-Lab/VideoFeedback-Bench"
bench_name="vbench"
frames_dir="../data/${bench_name}/test"
name_postfixs="[${bench_name}]"
result_file="./eval_results/${bench_name}/eval_${bench_name}_${metric_name}.json"


python eval_feature_metric.py --model_repo_name $model_repo_name \
    --data_repo_name $data_repo_name \
    --metric_name $metric_name \
    --bench_name $bench_name \
    --frames_dir $frames_dir  \
    --name_postfixs $name_postfixs \
    --result_file $result_file